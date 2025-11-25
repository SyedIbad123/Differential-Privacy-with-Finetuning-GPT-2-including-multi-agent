"""
DP LoRA Fine-tuning Script — ACI-Bench (Healthcare) Edition
- Uses single CSV (id, subset/dataset, dialogue, note, aug_note/augmented note)
- Prefers augmented note as target when available
- Clinical-aware augmentation (input-side only)
- Manual CE loss with 'mean' reduction (Opacus-safe)
- Mild DP noise for tiny clinical datasets
- Early stopping + weight decay
"""

import os
import math
import json
import random
from datetime import datetime
from typing import Tuple, List, Dict, Any, Optional
import re

import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset as TorchDataset
from tqdm import tqdm
import matplotlib.pyplot as plt

# Hugging Face imports
from datasets import Dataset, load_dataset, concatenate_datasets
from transformers import (
    GPT2LMHeadModel,
    GPT2Tokenizer,
    get_linear_schedule_with_warmup
)

# Differential Privacy (Opacus)
from opacus import PrivacyEngine
from opacus.validators import ModuleValidator
from opacus.utils.batch_memory_manager import BatchMemoryManager

# Parameter-Efficient Fine-Tuning
from peft import LoraConfig, get_peft_model, TaskType

# Metrics for privacy eval
from sklearn.metrics import roc_auc_score, average_precision_score, roc_curve

class Config:
    """Central configuration for the pipeline"""
    DATA_SOURCE = "csv"                    
    CSV_PATH = "/content/aci_bench_refined.csv" 

    DATASET_NAME = "trendmicro-ailab/Primus-Instruct"

    TARGET_SIZE = 2000
    AUGMENT_METHOD = 'clinical_simple'
    MAX_LENGTH = 384

    MODEL_NAME = "gpt2"                  

    LORA_R = 16
    LORA_ALPHA = 32
    LORA_DROPOUT = 0.05
    LORA_TARGET_MODULES = ["c_attn", "c_proj", "mlp.c_fc", "mlp.c_proj"]
    USE_BIAS_TUNING = True

    BATCH_SIZE = 16                      
    MAX_PHYSICAL_BATCH_SIZE = 8
    LEARNING_RATE = 3e-5
    WEIGHT_DECAY = 0.01
    EPOCHS = 8
    WARMUP_RATIO = 0.06

    NOISE_MULTIPLIER = 1.0
    MAX_GRAD_NORM = 1.0
    TARGET_DELTA = 1e-5
    SECURE_RNG = False                  

    EARLY_STOP_PATIENCE = 2

    MIA_NUM_SAMPLES = 200
    EVAL_BATCH_SIZE = 16

    SEED = 42
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")



def set_seed(seed: int):
    """Set all random seeds for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def ensure_dir(path: str):
    """Create directory if it doesn't exist"""
    os.makedirs(path, exist_ok=True)

def get_timestamp():
    """Get timestamp for experiment tracking"""
    return datetime.now().strftime("%Y%m%d_%H%M%S")

def clear_gpu_memory():
    """Clear GPU memory cache"""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        import gc
        gc.collect()


def dephi(text: str) -> str:
    """Light PHI redaction (names/phones/emails/dates) -> placeholders."""
    if not isinstance(text, str):
        return ""
    text = re.sub(r'\b(?:Dr\.?|Doctor)\s+[A-Z][a-z]+\b', 'PROVIDER_X', text)
    text = re.sub(r'\b(Mr\.?|Ms\.?|Mrs\.?)\s+[A-Z][a-z]+\b', 'PATIENT_X', text)
    text = re.sub(r'\b\d{3}[-\s]?\d{3}[-\s]?\d{4}\b', 'PHONE_X', text)
    text = re.sub(r'[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}', 'EMAIL_X', text)
    text = re.sub(r'\b(?:\d{1,2}[/-]\d{1,2}[/-]\d{2,4}|(?:19|20)\d{2}-\d{2}-\d{2})\b', 'DATE_X', text)
    return text

def row_to_messages(row: Dict[str, Any]) -> Dict[str, Any]:
    """
    Turn one CSV row into expected {'messages': [...]}
    Prefer augmented note if present: aug_note / augmented note -> fallback to note
    """
    dialogue_raw = row.get("dialogue", "") or row.get("dialog", "") or ""
    note_raw = row.get("aug_note", "") or row.get("augmented note", "") or row.get("note", "") or ""

    dialogue = dephi(str(dialogue_raw).strip())
    note = dephi(str(note_raw).strip())

    return {
        "messages": [
            {"role": "system", "content": "You are a clinical scribe generating concise SOAP notes."},
            {"role": "user", "content": f"<DIALOGUE>\n{dialogue}\n</DIALOGUE>\nWrite a SOAP note."},
            {"role": "assistant", "content": f"<SOAP>\n{note}\n</SOAP>"}
        ]
    }

class ConsistentCrossEntropyLoss(nn.Module):
    """CrossEntropyLoss with guaranteed 'mean' reduction for Opacus compatibility"""

    def __init__(self):
        super().__init__()
        self.criterion = nn.CrossEntropyLoss(ignore_index=-100, reduction='mean')

    def forward(self, logits, labels):
        """
        Compute loss for causal LM
        logits: (batch, seq_len, vocab_size)
        labels: (batch, seq_len) with -100 for ignored positions
        """
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()

        batch_size, seq_len, vocab_size = shift_logits.shape
        flat_logits = shift_logits.view(-1, vocab_size)
        flat_labels = shift_labels.view(-1)

        return self.criterion(flat_logits, flat_labels)


class DataAugmenter:
    """Handles data augmentation strategies (clinical mode)"""

    def __init__(self, method='clinical_simple'):
        self.method = method

    def augment_dataset(self, original_dataset, target_size):
        print(f"\n{'='*70}")
        print(f"DATA AUGMENTATION")
        print(f"{'='*70}")
        print(f"Original size: {len(original_dataset)}")
        print(f"Target size: {target_size}")

        if len(original_dataset) >= target_size:
            print("Dataset already sufficient!")
            return original_dataset

        num_to_generate = target_size - len(original_dataset)
        print(f"Generating {num_to_generate} augmented samples...")
        return self._augment_clinical_simple(original_dataset, num_to_generate)

    def _augment_clinical_simple(self, original_dataset, num_to_generate):
        """
        Simple clinical-safe aug:
        - Add SOAP-friendly prefixes on assistant note (occasionally)
        - Minor paraphrases / ASR-like artifacts on user dialogue
        - Light truncation on dialogue (avoid cutting mid-token)
        """
        rng = random.Random(Config.SEED)
        augmented = []

        soap_prefixes = ["Subjective: ", "Objective: ", "Assessment: ", "Plan: "]
        for i in tqdm(range(num_to_generate), desc="Augmenting"):
            base = original_dataset[i % len(original_dataset)]
            msgs = base['messages']

            new_messages = []
            for m in msgs:
                role = m.get('role', '')
                content = m.get('content', '')

                if role == 'assistant' and rng.random() < 0.3:
                    content = rng.choice(soap_prefixes) + content

                if role == 'user' and rng.random() < 0.2 and len(content) > 60:
                    content = content.replace(" the ", " tha ").replace(" with ", " w/ ")

                if role == 'user' and rng.random() < 0.15 and len(content) > 120:
                    cut = rng.randrange(100, min(len(content), 220))
                    content = content[:cut].rsplit(' ', 1)[0] + "..."

                new_messages.append({"role": role, "content": content})

            augmented.append({"messages": new_messages})

        aug = Dataset.from_list(augmented)
        combined = concatenate_datasets([original_dataset, aug]).shuffle(seed=Config.SEED)
        print(f"✓ Augmentation complete! Final size: {len(combined)}")
        return combined

class DataProcessor:
    """Handles data loading and preprocessing"""

    def __init__(self, tokenizer, max_length=Config.MAX_LENGTH):
        self.tokenizer = tokenizer
        self.max_length = max_length

    def load_and_preprocess(self, dataset_dict):
        """Load and preprocess dataset"""
        ds = dataset_dict['train']

        processed_data = []
        for item in tqdm(ds, desc="Processing data"):
            msgs = item['messages']
            parts = []
            for m in msgs:
                role = m.get('role', '')
                content = m.get('content', '').strip()
                parts.append(f"<<{role.upper()}>>\n{content}\n<</{role.upper()}>>")
            conversation = "\n\n".join(parts) + self.tokenizer.eos_token
            processed_data.append(conversation)

        rng = np.random.default_rng(Config.SEED)
        rng.shuffle(processed_data)

        n = len(processed_data)
        train_size = int(0.8 * n)
        val_size = int(0.1 * n)

        train_texts = processed_data[:train_size]
        val_texts = processed_data[train_size:train_size + val_size]
        test_texts = processed_data[train_size + val_size:]

        train_data = TextDataset(train_texts, self.tokenizer, self.max_length)
        val_data = TextDataset(val_texts, self.tokenizer, self.max_length)
        test_data = TextDataset(test_texts, self.tokenizer, self.max_length)

        return train_data, val_data, test_data

class TextDataset(TorchDataset):
    """Custom dataset for text data"""

    def __init__(self, texts, tokenizer, max_length):
        self.texts = texts
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )

        input_ids = encoding['input_ids'].squeeze()
        attention_mask = encoding['attention_mask'].squeeze()
        labels = input_ids.clone()
        labels[attention_mask == 0] = -100

        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'labels': labels
        }



def load_aci_csv(csv_path: str) -> Dataset:
    """
    Load ACI-Bench CSV and convert to HF Dataset with expected `messages` field.
    Handles flexible column names (subset/dataset, id/encounter_id, aug_note/augmented note).
    """
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"CSV not found at {csv_path}")

    raw = load_dataset("csv", data_files={"train": csv_path})
    mapped = raw["train"].map(
        lambda r: row_to_messages(r),
        remove_columns=raw["train"].column_names
    )
    return mapped


def load_and_prepare_model(model_name, tokenizer):
    """Load base model, apply LoRA, and prepare for DP training"""
    print(f"\n{'='*70}")
    print(f"MODEL SETUP")
    print(f"{'='*70}")

    print(f"Loading {model_name}...")
    model = GPT2LMHeadModel.from_pretrained(model_name)
    model.resize_token_embeddings(len(tokenizer))

    print("Applying LoRA configuration...")
    lora_config = LoraConfig(
        r=Config.LORA_R,
        lora_alpha=Config.LORA_ALPHA,
        lora_dropout=Config.LORA_DROPOUT,
        target_modules=Config.LORA_TARGET_MODULES,
        task_type=TaskType.CAUSAL_LM,
        bias="all" if Config.USE_BIAS_TUNING else "none",
        fan_in_fan_out=True 
    )
    model = get_peft_model(model, lora_config)

    if Config.USE_BIAS_TUNING:
        print("Enabling bias-only training...")
        for name, param in model.named_parameters():
            if ('bias' in name) or ('lora' in name):
                param.requires_grad = True
            else:
                param.requires_grad = False

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    print(f"Trainable ratio: {trainable_params/total_params:.2%}")

    print("Making model DP-compatible...")
    model = ModuleValidator.fix(model)

    return model


class EarlyStopper:
    def __init__(self, patience=2):
        self.patience = patience
        self.best = float("inf")
        self.count = 0

    def step(self, val_loss: float) -> bool:
        if val_loss + 1e-6 < self.best:
            self.best = val_loss
            self.count = 0
            return False
        self.count += 1
        return self.count > self.patience

class DPTrainer:
    """Differential Privacy trainer with manual loss computation"""

    def __init__(self, model, train_loader, val_loader, optimizer, privacy_engine, scheduler, device):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.optimizer = optimizer
        self.privacy_engine = privacy_engine
        self.scheduler = scheduler
        self.device = device
        self.criterion = ConsistentCrossEntropyLoss()

        self.train_losses = []
        self.val_losses = []
        self.train_perplexities = []
        self.val_perplexities = []
        self.privacy_epsilons = []
        self.early_stopper = EarlyStopper(patience=Config.EARLY_STOP_PATIENCE)

    def compute_loss(self, batch):
        """Compute loss manually to ensure consistent reduction"""
        input_ids = batch['input_ids'].to(self.device)
        attention_mask = batch['attention_mask'].to(self.device)
        labels = batch['labels'].to(self.device)

        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=True
        )
        logits = outputs.logits

        loss = self.criterion(logits, labels)
        return loss

    def train_epoch(self):
        """Train for one epoch"""
        self.model.train()
        total_loss = 0.0
        num_batches = 0

        with BatchMemoryManager(
            data_loader=self.train_loader,
            max_physical_batch_size=Config.MAX_PHYSICAL_BATCH_SIZE,
            optimizer=self.optimizer
        ) as memory_safe_loader:

            pbar = tqdm(memory_safe_loader, desc="Training", leave=False)
            for batch in pbar:
                loss = self.compute_loss(batch)

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                if self.scheduler is not None:
                    self.scheduler.step()

                total_loss += loss.item()
                num_batches += 1

                pbar.set_postfix({
                    'loss': f'{loss.item():.4f}',
                    'lr': f'{self.scheduler.get_last_lr()[0]:.2e}' if self.scheduler else 'n/a'
                })

        avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
        perplexity = math.exp(min(avg_loss, 100))
        return avg_loss, perplexity

    def evaluate(self):
        """Evaluate model on validation set"""
        self.model.eval()
        total_loss = 0.0
        num_batches = 0

        with torch.no_grad():
            for batch in tqdm(self.val_loader, desc="Evaluating", leave=False):
                loss = self.compute_loss(batch)
                total_loss += loss.item()
                num_batches += 1

        avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
        perplexity = math.exp(min(avg_loss, 100))
        return avg_loss, perplexity

    def train(self, epochs):
        """Main training loop"""
        print(f"\n{'='*70}")
        print("TRAINING WITH DIFFERENTIAL PRIVACY")
        print(f"{'='*70}")
        print(f"Target δ: {Config.TARGET_DELTA}")
        stop_early = False
        for epoch in range(epochs):
            if stop_early:
                break

            print(f"\nEpoch {epoch+1}/{epochs}")
            train_loss, train_ppl = self.train_epoch()
            self.train_losses.append(train_loss)
            self.train_perplexities.append(train_ppl)

            val_loss, val_ppl = self.evaluate()
            self.val_losses.append(val_loss)
            self.val_perplexities.append(val_ppl)

            try:
                epsilon = self.privacy_engine.get_epsilon(Config.TARGET_DELTA)
            except Exception:
                epsilon = float('nan')
            self.privacy_epsilons.append(epsilon)

            print(f"  Train Loss: {train_loss:.4f} | Perplexity: {train_ppl:.2f}")
            print(f"  Val   Loss: {val_loss:.4f} | Perplexity: {val_ppl:.2f}")
            print(f"  Privacy budget (ε): {epsilon if not math.isnan(epsilon) else 'N/A'}")

            if self.early_stopper.step(val_loss):
                print(f"Early stopping triggered (patience={Config.EARLY_STOP_PATIENCE}).")
                stop_early = True

            if epoch % 2 == 0:
                clear_gpu_memory()

        return {
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'train_perplexities': self.train_perplexities,
            'val_perplexities': self.val_perplexities,
            'privacy_epsilons': self.privacy_epsilons
        }



class PrivacyEvaluator:
    """Evaluate privacy using membership inference attacks"""

    def __init__(self, model, tokenizer, device):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.criterion = ConsistentCrossEntropyLoss()

    def compute_loss(self, text):
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=Config.MAX_LENGTH,
            return_tensors='pt'
        )
        input_ids = encoding['input_ids'].to(self.device)
        attention_mask = encoding['attention_mask'].to(self.device)
        labels = input_ids.clone()
        labels[attention_mask == 0] = -100

        with torch.no_grad():
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                return_dict=True
            )
            logits = outputs.logits
            loss = self.criterion(logits, labels)
        return loss.item()

    def membership_inference_attack(self, train_data, test_data, num_samples=500):
        print(f"\n{'='*70}")
        print("MEMBERSHIP INFERENCE ATTACK EVALUATION")
        print(f"{'='*70}")
        self.model.eval()

        num_samples = min(num_samples, len(train_data), len(test_data))
        train_indices = random.sample(range(len(train_data)), num_samples)
        test_indices = random.sample(range(len(test_data)), num_samples)

        train_losses = []
        for idx in tqdm(train_indices, desc="MIA train losses"):
            text = train_data.texts[idx]
            train_losses.append(self.compute_loss(text))

        test_losses = []
        for idx in tqdm(test_indices, desc="MIA test losses"):
            text = test_data.texts[idx]
            test_losses.append(self.compute_loss(text))

        y_true = [1] * len(train_losses) + [0] * len(test_losses)
        scores = [-l for l in train_losses + test_losses]

        auc = roc_auc_score(y_true, scores)
        ap = average_precision_score(y_true, scores)
        threshold = np.median(scores)
        predictions = [1 if s > threshold else 0 for s in scores]
        accuracy = sum(p == t for p, t in zip(predictions, y_true)) / len(y_true)

        results = {
            'auc': auc,
            'average_precision': ap,
            'accuracy': accuracy,
            'train_losses': train_losses,
            'test_losses': test_losses
        }

        print(f"\n✓ MIA Results: ROC AUC={auc:.3f}, AP={ap:.3f}, Acc={accuracy*100:.1f}%")
        return results


class Visualizer:
    """Create training and evaluation visualizations"""

    @staticmethod
    def plot_training_curves(metrics, output_dir):
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))

        axes[0, 0].plot(metrics['train_losses'], label='Train', marker='o')
        axes[0, 0].plot(metrics['val_losses'], label='Validation', marker='s')
        axes[0, 0].set_xlabel('Epoch'); axes[0, 0].set_ylabel('Loss'); axes[0, 0].legend(); axes[0, 0].grid(True, alpha=0.3)

        axes[0, 1].plot(metrics['train_perplexities'], label='Train', marker='o')
        axes[0, 1].plot(metrics['val_perplexities'], label='Validation', marker='s')
        axes[0, 1].set_xlabel('Epoch'); axes[0, 1].set_ylabel('Perplexity'); axes[0, 1].legend(); axes[0, 1].grid(True, alpha=0.3)

        axes[1, 0].plot(metrics['privacy_epsilons'], marker='o')
        axes[1, 0].set_xlabel('Epoch'); axes[1, 0].set_ylabel('Privacy Budget (ε)'); axes[1, 0].grid(True, alpha=0.3)

        axes[1, 1].plot(metrics['privacy_epsilons'], metrics['val_perplexities'], marker='o')
        axes[1, 1].set_xlabel('Privacy Budget (ε)'); axes[1, 1].set_ylabel('Validation Perplexity'); axes[1, 1].grid(True, alpha=0.3)

        plt.tight_layout()
        os.makedirs(output_dir, exist_ok=True)
        plt.savefig(os.path.join(output_dir, 'training_curves.png'), dpi=300, bbox_inches='tight')
        plt.close()
        print(f"✓ Training curves saved to {output_dir}/training_curves.png")

    @staticmethod
    def plot_mia_results(mia_results, output_dir):
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))

        train_losses = mia_results['train_losses']; test_losses = mia_results['test_losses']
        axes[0].hist(train_losses, bins=30, alpha=0.5, label='Train'); axes[0].hist(test_losses, bins=30, alpha=0.5, label='Test')
        axes[0].set_xlabel('Loss'); axes[0].set_ylabel('Count'); axes[0].legend(); axes[0].grid(True, alpha=0.3)

        y_true = [1] * len(train_losses) + [0] * len(test_losses); scores = [-l for l in train_losses + test_losses]
        fpr, tpr, _ = roc_curve(y_true, scores)
        axes[1].plot(fpr, tpr, label=f'ROC (AUC={mia_results["auc"]:.3f})'); axes[1].plot([0,1],[0,1],'k--')
        axes[1].set_xlabel('FPR'); axes[1].set_ylabel('TPR'); axes[1].legend(); axes[1].grid(True, alpha=0.3)

        plt.tight_layout()
        os.makedirs(output_dir, exist_ok=True)
        plt.savefig(os.path.join(output_dir, 'mia_results.png'), dpi=300, bbox_inches='tight')
        plt.close()
        print(f"✓ MIA results saved to {output_dir}/mia_results.png")


def setup_differential_privacy(model, optimizer, train_loader, sample_size):
    """Set up differential privacy (hooks preferred; secure_mode configurable)"""
    privacy_engine = PrivacyEngine(secure_mode=Config.SECURE_RNG)

    try:
        model, optimizer, train_loader = privacy_engine.make_private(
            module=model,
            optimizer=optimizer,
            data_loader=train_loader,
            noise_multiplier=Config.NOISE_MULTIPLIER,
            max_grad_norm=Config.MAX_GRAD_NORM,
            batch_size=Config.BATCH_SIZE,
            sample_size=sample_size,
            loss_reduction='mean',
            grad_sample_mode='hooks'
        )
        print("✓ Privacy engine configured with 'hooks' mode")
    except Exception as e1:
        print(f"Approach 1 failed: {e1}")
        print("Trying alternative approach...")
        try:
            model, optimizer, train_loader = privacy_engine.make_private(
                module=model,
                optimizer=optimizer,
                data_loader=train_loader,
                noise_multiplier=Config.NOISE_MULTIPLIER,
                max_grad_norm=Config.MAX_GRAD_NORM,
                batch_size=Config.BATCH_SIZE,
                sample_size=sample_size,
                grad_sample_mode='hooks'
            )
            print("✓ Privacy engine configured without explicit loss_reduction")
        except Exception as e2:
            print(f"Approach 2 failed: {e2}")
            print("Trying final approach with 'ew' mode...")
            model, optimizer, train_loader = privacy_engine.make_private(
                module=model,
                optimizer=optimizer,
                data_loader=train_loader,
                noise_multiplier=Config.NOISE_MULTIPLIER,
                max_grad_norm=Config.MAX_GRAD_NORM,
                batch_size=Config.BATCH_SIZE,
                sample_size=sample_size,
                grad_sample_mode='ew'
            )
            print("✓ Privacy engine configured with 'ew' mode")

    return model, optimizer, train_loader, privacy_engine



def main():
    set_seed(Config.SEED)
    timestamp = get_timestamp()
    output_dir = f"runs/dp_lora_{timestamp}"
    ensure_dir(output_dir)

    print("\n" + "="*70)
    print("DIFFERENTIAL PRIVACY FINE-TUNING WITH LoRA (ACI-Bench)")
    print("="*70)
    print(f"Output directory: {output_dir}")
    print(f"Device: {Config.DEVICE}")

    config_dict = {
        'data_source': Config.DATA_SOURCE,
        'csv_path': Config.CSV_PATH,
        'dataset_fallback': Config.DATASET_NAME,
        'target_size': Config.TARGET_SIZE,
        'model': Config.MODEL_NAME,
        'lora_r': Config.LORA_R,
        'lora_alpha': Config.LORA_ALPHA,
        'batch_size': Config.BATCH_SIZE,
        'learning_rate': Config.LEARNING_RATE,
        'weight_decay': Config.WEIGHT_DECAY,
        'epochs': Config.EPOCHS,
        'noise_multiplier': Config.NOISE_MULTIPLIER,
        'max_grad_norm': Config.MAX_GRAD_NORM,
        'target_delta': Config.TARGET_DELTA,
        'use_bias_tuning': Config.USE_BIAS_TUNING,
        'secure_rng': Config.SECURE_RNG,
        'timestamp': timestamp
    }
    with open(os.path.join(output_dir, 'config.json'), 'w') as f:
        json.dump(config_dict, f, indent=2)

    print(f"\n{'='*70}")
    print("STEP 1: DATA LOADING AND AUGMENTATION")
    print(f"{'='*70}")
    try:
        if Config.DATA_SOURCE == "csv":
            base_ds = load_aci_csv(Config.CSV_PATH)  
            augmenter = DataAugmenter(method=Config.AUGMENT_METHOD)
            augmented = augmenter.augment_dataset(base_ds, Config.TARGET_SIZE)
        else:
            raw_ds = load_dataset(Config.DATASET_NAME)
            augmenter = DataAugmenter(method=Config.AUGMENT_METHOD)
            augmented = augmenter.augment_dataset(raw_ds['train'], Config.TARGET_SIZE)

    except Exception as e:
        print(f"Error loading dataset: {e}")
        print("Creating dummy dataset for testing and augmenting...")
        dummy = [{'messages': [
            {'role': 'system', 'content': 'You are a clinical scribe generating concise SOAP notes.'},
            {'role': 'user', 'content': f'<DIALOGUE>\nSample {i} with cough and fever\n</DIALOGUE>\nWrite a SOAP note.'},
            {'role': 'assistant', 'content': f'<SOAP>\nAssessment: viral URI\nPlan: rest, fluids\n</SOAP>'}
        ]} for i in range(100)]
        base = Dataset.from_list(dummy)
        augmenter = DataAugmenter(method=Config.AUGMENT_METHOD)
        augmented = augmenter.augment_dataset(base, Config.TARGET_SIZE)

    print(f"Dataset size after augmentation: {len(augmented)}")


    print(f"\n{'='*70}")
    print("STEP 2: TOKENIZATION AND DATA PROCESSING")
    print(f"{'='*70}")
    tokenizer = GPT2Tokenizer.from_pretrained(Config.MODEL_NAME)
    tokenizer.pad_token = tokenizer.eos_token

    processor = DataProcessor(tokenizer, Config.MAX_LENGTH)
    train_data, val_data, test_data = processor.load_and_preprocess({'train': augmented})

    print(f"✓ Data splits: train={len(train_data)}, val={len(val_data)}, test={len(test_data)}")

    g = torch.Generator()
    g.manual_seed(Config.SEED)
    train_loader = DataLoader(
        train_data,
        batch_size=Config.BATCH_SIZE,
        shuffle=True,
        drop_last=True,
        generator=g
    )
    val_loader = DataLoader(
        val_data,
        batch_size=Config.EVAL_BATCH_SIZE,
        shuffle=False
    )


    model = load_and_prepare_model(Config.MODEL_NAME, tokenizer)
    model = model.to(Config.DEVICE)

 
    optimizer = AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=Config.LEARNING_RATE,
        weight_decay=Config.WEIGHT_DECAY
    )

    num_training_steps = len(train_loader) * Config.EPOCHS
    num_warmup_steps = int(num_training_steps * Config.WARMUP_RATIO)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=num_warmup_steps, num_training_steps=num_training_steps)

    print(f"✓ Optimizer: AdamW (lr={Config.LEARNING_RATE}, weight_decay={Config.WEIGHT_DECAY})")
    print(f"✓ Scheduler steps: total={num_training_steps}, warmup={num_warmup_steps}")

  
    print(f"\n{'='*70}")
    print("STEP 3: SETUP DIFFERENTIAL PRIVACY")
    print(f"{'='*70}")

    sample_size = len(train_data)
    print(f"Sample size for DP: {sample_size} | Batch size: {Config.BATCH_SIZE}")
    model, optimizer, train_loader, privacy_engine = setup_differential_privacy(
        model, optimizer, train_loader, sample_size
    )


    trainer = DPTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        privacy_engine=privacy_engine,
        scheduler=scheduler,
        device=Config.DEVICE
    )

 
    training_metrics = trainer.train(Config.EPOCHS)

    model_save_path = os.path.join(output_dir, 'dp_lora_model')
    os.makedirs(model_save_path, exist_ok=True)

    if hasattr(model, '_module'):
        model._module.save_pretrained(model_save_path)
    else:
        model.save_pretrained(model_save_path)

    tokenizer.save_pretrained(model_save_path)
    print(f"✓ Model saved to {model_save_path}")

    print("\n" + "="*70)
    print("MODEL SAVE VERIFICATION")
    print("="*70)
    if os.path.exists(model_save_path):
        saved_files = os.listdir(model_save_path)
        print(f"✓ Model directory exists: {model_save_path}")
        print(f"✓ Files saved ({len(saved_files)} files):")
        for f in sorted(saved_files):
            file_path = os.path.join(model_save_path, f)
            size = os.path.getsize(file_path) / (1024 * 1024)  
            print(f"  - {f} ({size:.2f} MB)")

        required_files = ['adapter_config.json', 'adapter_model.bin', 'tokenizer_config.json']
        missing_files = []
        for req in required_files:
            if req not in saved_files and req.replace('.bin', '.safetensors') not in saved_files:
                missing_files.append(req)

        if missing_files:
            print(f"\n⚠ WARNING: Missing required files: {missing_files}")
        else:
            print(f"\n✓ All critical LoRA adapter files present!")
    else:
        print(f"✗ ERROR: Model directory does not exist: {model_save_path}")

 
    print(f"\n{'='*70}")
    print("STEP 4: PRIVACY EVALUATION (MIA)")
    print(f"{'='*70}")

    eval_model = model._module if hasattr(model, '_module') else model
    evaluator = PrivacyEvaluator(eval_model, tokenizer, Config.DEVICE)
    mia_results = evaluator.membership_inference_attack(
        train_data, test_data,
        num_samples=min(Config.MIA_NUM_SAMPLES, len(train_data), len(test_data))
    )

   
    visualizer = Visualizer()
    visualizer.plot_training_curves(training_metrics, output_dir)
    visualizer.plot_mia_results(mia_results, output_dir)

   
    final_epsilon = training_metrics['privacy_epsilons'][-1] if training_metrics['privacy_epsilons'] else float('nan')
    results_summary = {
        'config': config_dict,
        'training': {
            'final_train_loss': training_metrics['train_losses'][-1] if training_metrics['train_losses'] else None,
            'final_val_loss': training_metrics['val_losses'][-1] if training_metrics['val_losses'] else None,
            'final_train_perplexity': training_metrics['train_perplexities'][-1] if training_metrics['train_perplexities'] else None,
            'final_val_perplexity': training_metrics['val_perplexities'][-1] if training_metrics['val_perplexities'] else None,
            'final_epsilon': final_epsilon,
            'target_delta': Config.TARGET_DELTA
        },
        'privacy_evaluation': {
            'mia_auc': mia_results['auc'],
            'mia_average_precision': mia_results['average_precision'],
            'mia_accuracy': mia_results['accuracy']
        }
    }
    with open(os.path.join(output_dir, 'results_summary.json'), 'w') as f:
        json.dump(results_summary, f, indent=2)

    
    print(f"\n{'='*70}")
    print("FINAL RESULTS SUMMARY")
    print(f"{'='*70}")
    if training_metrics['val_losses']:
        print(f"  Final Validation Loss: {training_metrics['val_losses'][-1]:.4f}")
        print(f"  Final Validation Perplexity: {training_metrics['val_perplexities'][-1]:.2f}")
    print(f"  Final ε: {final_epsilon if not math.isnan(final_epsilon) else 'N/A'}")
    print(f"  Target δ: {Config.TARGET_DELTA}")
    print(f"  MIA ROC AUC: {mia_results['auc']:.3f}")
    print(f"\nOutputs saved to: {output_dir}")

    print("\n" + "="*70)
    print("TO USE THIS MODEL IN agent_with_dp.py:")
    print("="*70)
    print(f"Set this path in your Colab notebook:")
    print(f"  os.environ['LORA_DIR'] = '{model_save_path}'")
    print("\nOr copy this exact path:")
    print(f"  {model_save_path}")
    print("="*70)

    return results_summary, eval_model, tokenizer, model_save_path


if __name__ == "__main__":
    try:
        results, model, tokenizer, model_path = main()
        print("\n✓ Experiment completed successfully!")
        print(f"✓ Model ready at: {model_path}")
    except Exception as e:
        print(f"\n❌ Error during execution: {e}")
        import traceback
        traceback.print_exc()
    finally:
        clear_gpu_memory()
