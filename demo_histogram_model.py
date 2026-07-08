#!/usr/bin/env python3
"""
Histogram Model Demo Script
===========================

This script demonstrates the complete histogram model pipeline with clear,
understandable examples using the same dummy patient data from demo_histogram_dataset.py.

It shows:
1. How the Vector Quantizer works on histogram windows (Phase 1 - Quantizer Training)
2. How the Autoregressive Model works with gap tokens (Phase 2 - AR Training)
3. The complete vocabulary structure (quantized tokens + gap tokens + special tokens)
4. Real examples with Patient 1 (regular visits) and Patient 2 (sparse with gaps)

Key Features:
- Uses the same patient data as the histogram dataset demo
- Shows quantizer training mode (EmptyWindowMode.IGNORE - no gaps)
- Shows autoregressive training mode (EmptyWindowMode.SINGLE_GAP - with gap tokens)
- Explains the complete vocabulary mapping clearly
- Demonstrates how Patient 2 gets gap tokens for the 13-month gap
"""

import sys
from datetime import datetime, timedelta

import numpy as np
import polars as pl
import torch

# Add src to path
sys.path.insert(0, "src")

from morgen.data.histogram_sequence_processor import HistogramSequenceProcessor
from morgen.model.histogram_model import GapTokenConfig, HistogramModel, ModelMode
from morgen.model.simplified_quantizer import SimplifiedAutoencoderVQ


def create_dummy_patient_data():
    """Create the same dummy patient data as in demo_histogram_dataset.py."""

    print("🏥 CREATING DUMMY PATIENT DATA")
    print("=" * 60)

    # Patient 1: Regular visits over 3 months
    patient1_data = [
        # January visits - Dense cluster
        {"subject_id": 1, "time": datetime(2020, 1, 5), "code": 10},  # Diagnosis code
        {"subject_id": 1, "time": datetime(2020, 1, 5), "code": 20},  # Lab code
        {"subject_id": 1, "time": datetime(2020, 1, 12), "code": 30},  # Treatment code
        # February visits - Regular follow-up
        {"subject_id": 1, "time": datetime(2020, 2, 3), "code": 10},  # Same diagnosis
        {"subject_id": 1, "time": datetime(2020, 2, 15), "code": 40},  # New medication
        # March visits - Continued care
        {"subject_id": 1, "time": datetime(2020, 3, 10), "code": 50},  # Follow-up
        {"subject_id": 1, "time": datetime(2020, 3, 20), "code": 20},  # Repeat lab
    ]

    # Patient 2: Sparse visits with LONG gap (13 months!)
    patient2_data = [
        # January 2020 visits (clustered emergency)
        {"subject_id": 2, "time": datetime(2020, 1, 8), "code": 60},  # Emergency
        {"subject_id": 2, "time": datetime(2020, 1, 8), "code": 70},  # Treatment
        {"subject_id": 2, "time": datetime(2020, 1, 9), "code": 80},  # Discharge
        # LONG GAP - 13 months of no visits (Jan 2020 -> Mar 2021)
        # March 2021 visits (13 months later!)
        {"subject_id": 2, "time": datetime(2021, 3, 25), "code": 60},  # Follow-up
    ]

    all_data = patient1_data + patient2_data
    df = pl.DataFrame(all_data)

    print("📊 Dataset Summary:")
    print(f"   - Patients: {df['subject_id'].n_unique()}")
    print(f"   - Total events: {len(df)}")
    print(f"   - Date range: {df['time'].min()} to {df['time'].max()}")
    print(f"   - Unique medical codes: {sorted(df['code'].unique())}")

    # Show each patient's timeline
    for patient_id in sorted(df["subject_id"].unique()):
        patient_data = df.filter(pl.col("subject_id") == patient_id)

        print(f"\n👤 PATIENT {patient_id} TIMELINE:")
        print("-" * 30)

        for row in patient_data.iter_rows(named=True):
            print(f"   {row['time'].strftime('%Y-%m-%d')}: Medical Code {row['code']}")

        # Calculate timeline span
        start_date = patient_data["time"].min()
        end_date = patient_data["time"].max()
        span_days = (end_date - start_date).days
        print(f"   Timeline span: {span_days} days ({span_days / 30:.1f} months)")

    return df


def create_mock_histogram_windows(patient_data: pl.DataFrame, window_size_days: int = 30):
    """Create mock histogram windows from patient data (simplified version)."""

    # Group by patient
    windows_by_patient = {}

    for patient_id in sorted(patient_data["subject_id"].unique()):
        patient_events = patient_data.filter(pl.col("subject_id") == patient_id)

        # Get date range
        start_date = patient_events["time"].min()
        end_date = patient_events["time"].max()

        # Create 30-day windows
        windows = []
        current_date = start_date

        while current_date <= end_date:
            window_end = current_date + timedelta(days=window_size_days)

            # Find events in this window
            window_events = patient_events.filter(
                (pl.col("time") >= current_date) & (pl.col("time") < window_end)
            )

            if len(window_events) > 0:
                # Create binary histogram (codes 10-80 map to indices 0-7)
                histogram = torch.zeros(8)  # 8 possible codes (10,20,30,40,50,60,70,80)

                for code in window_events["code"].to_list():
                    code_idx = (code - 10) // 10  # Map 10->0, 20->1, ..., 80->7
                    histogram[code_idx] = 1.0  # Binary presence

                windows.append(
                    {
                        "start_date": current_date,
                        "end_date": window_end,
                        "histogram": histogram,
                        "codes": window_events["code"].to_list(),
                        "is_gap": False,
                    }
                )
            else:
                # Empty window (gap)
                windows.append(
                    {
                        "start_date": current_date,
                        "end_date": window_end,
                        "histogram": torch.zeros(8),
                        "codes": [],
                        "is_gap": True,
                    }
                )

            current_date = window_end

        windows_by_patient[patient_id] = windows

    return windows_by_patient


def demonstrate_vector_quantizer_phase1(windows_by_patient: dict):
    """Demonstrate Phase 1: Vector Quantizer Training (IGNORE empty windows)."""

    print("\n🔧 PHASE 1: VECTOR QUANTIZER TRAINING")
    print("=" * 60)
    print("Mode: EmptyWindowMode.IGNORE (Skip empty windows)")
    print("Goal: Learn to compress binary histograms into discrete tokens")

    # Create simplified VQ model with small vocab for demo
    quantizer = SimplifiedAutoencoderVQ(
        vocab_size=8,  # 8 medical codes
        embedding_dim=4,  # Small latent space for demo
        n_embeddings=16,  # Small codebook for demo
        encoder_hidden_dims=[8, 6],
        decoder_hidden_dims=[6, 8],
    )

    print("\n📋 Vector Quantizer Configuration:")
    print(f"   - Input vocab size: {quantizer.vocab_size} medical codes")
    print(f"   - Embedding dimension: {quantizer.embedding_dim}")
    print(f"   - Codebook size: {quantizer.quantizer.n_embeddings} discrete tokens")
    print(f"   - VQ Beta (commitment): {quantizer.quantizer.beta}")

    # Process each patient
    for patient_id, windows in windows_by_patient.items():
        print(f"\n👤 PATIENT {patient_id} - Vector Quantizer Processing:")
        print("-" * 40)

        # Filter out empty windows (IGNORE mode)
        non_empty_windows = [w for w in windows if not w["is_gap"]]

        print(f"Original windows: {len(windows)}")
        print(f"Non-empty windows: {len(non_empty_windows)} (empty windows ignored)")

        if len(non_empty_windows) == 0:
            print("❌ No non-empty windows for quantizer training!")
            continue

        # Stack histograms for batch processing
        histograms = torch.stack([w["histogram"] for w in non_empty_windows])
        print(f"Histogram batch shape: {histograms.shape}")

        # Forward pass through quantizer
        with torch.no_grad():
            result = quantizer(histograms)

        print("\n🎯 Quantizer Results:")
        print(f"   - VQ Loss: {result['vq_loss'].item():.4f}")
        print(f"   - Reconstruction Loss: {result['reconstruction_loss'].item():.4f}")
        print(f"   - Total Loss: {result['total_loss'].item():.4f}")

        # Show detailed window-by-window results
        print("\n📊 Window-by-Window Quantization:")
        indices = result["indices"].tolist()

        for i, (window, token_idx) in enumerate(zip(non_empty_windows, indices, strict=False)):
            codes_str = ", ".join([f"Code{c}" for c in window["codes"]])
            start_date = window["start_date"].strftime("%Y-%m-%d")

            # Show which histogram elements are active
            active_elements = torch.nonzero(window["histogram"]).flatten()
            hist_str = ", ".join([f"[{idx.item()}]=1" for idx in active_elements])

            print(f"   Window {i + 1} ({start_date}): {codes_str}")
            print(f"      → Binary histogram: {hist_str}")
            print(f"      → Quantized token: {token_idx}")


def demonstrate_autoregressive_phase2(windows_by_patient: dict):
    """Demonstrate Phase 2: Autoregressive Training (WITH gap tokens)."""

    print("\n🤖 PHASE 2: AUTOREGRESSIVE MODEL TRAINING")
    print("=" * 60)
    print("Mode: EmptyWindowMode.SINGLE_GAP (Include gap tokens)")
    print("Goal: Learn sequences of [histogram_tokens, gap_tokens] for generation")

    # Create histogram model in AR mode
    model_config = {
        "num_hidden_layers": 2,
        "num_attention_heads": 2,
        "hidden_size": 32,
        "max_position_embeddings": 50,
        "vocab_size": 50,  # Will be updated
    }
    quantizer_config = {"vocab_size": 8, "embedding_dim": 4, "n_embeddings": 16}
    gap_config = GapTokenConfig(max_gap_length=1200)

    # Create autoregressive histogram model
    ar_model = HistogramModel(
        model_config=model_config,
        quantizer_config=quantizer_config,
        gap_token_config=gap_config,
        mode=ModelMode.AUTOREGRESSIVE,
    )

    print("\n📋 Autoregressive Model Configuration:")
    print(f"   - Quantizer tokens: 0-{ar_model.quantizer_vocab_size - 1} (histogram codes)")
    print(
        f"   - Gap tokens: {ar_model.gap_token_config.gap_token_start}-{ar_model.gap_token_config.gap_token_start + ar_model.gap_vocab_size - 1} (GAP_1, GAP_2, ..., GAP_{ar_model.gap_vocab_size})"
    )
    print(
        f"   - Special tokens: PAD={ar_model.special_tokens['PAD']}, BOS={ar_model.special_tokens['BOS']}, EOS={ar_model.special_tokens['EOS']}"
    )
    print(f"   - Total vocabulary: {ar_model.total_vocab_size} tokens")

    # Create sequence processor
    processor = HistogramSequenceProcessor(
        quantizer=ar_model.quantizer,
        vocab_size=quantizer_config["vocab_size"],
        max_gap_length=gap_config.max_gap_length,
        gap_token_strategy="multiple",
    )

    # Process each patient with gap handling
    for patient_id, windows in windows_by_patient.items():
        print(f"\n👤 PATIENT {patient_id} - Autoregressive Sequence Processing:")
        print("-" * 50)

        # Show the complete window timeline
        print(f"Complete timeline ({len(windows)} windows):")
        for i, window in enumerate(windows):
            start_date = window["start_date"].strftime("%Y-%m-%d")
            if window["is_gap"]:
                print(f"   Window {i + 1}: {start_date} - EMPTY (gap window)")
            else:
                codes = ", ".join([f"Code{c}" for c in window["codes"]])
                print(f"   Window {i + 1}: {start_date} - {codes}")

        # Create token sequence with gap handling
        tokens = []
        tokens.append(ar_model.special_tokens["BOS"])  # Start token

        # Process windows into tokens
        i = 0
        while i < len(windows):
            window = windows[i]

            if window["is_gap"]:
                # Count consecutive gaps
                gap_count = 0
                while i < len(windows) and windows[i]["is_gap"]:
                    gap_count += 1
                    i += 1

                # Add gap token(s)
                if gap_count > ar_model.gap_token_config.max_gap_length:
                    # Split large gaps into multiple tokens
                    remaining = gap_count
                    while remaining > 0:
                        current_gap = min(remaining, ar_model.gap_token_config.max_gap_length)
                        gap_token = ar_model.encode_gap_token(current_gap)
                        tokens.append(gap_token)
                        remaining -= current_gap
                        print(f"      → Added GAP_{current_gap} token: {gap_token}")
                else:
                    gap_token = ar_model.encode_gap_token(gap_count)
                    tokens.append(gap_token)
                    print(f"      → Added GAP_{gap_count} token: {gap_token}")
            else:
                # Process non-empty window -> quantize histogram
                histogram = window["histogram"].unsqueeze(0)  # Add batch dim

                with torch.no_grad():
                    # Encode -> quantize -> get discrete token
                    z_e = ar_model.quantizer.encode(histogram)
                    z_q, vq_loss, indices = ar_model.quantizer.quantize(z_e)
                    token_idx = indices[0].item()

                tokens.append(token_idx)
                codes = ", ".join([f"Code{c}" for c in window["codes"]])
                print(f"      → Histogram ({codes}) → Token: {token_idx}")
                i += 1

        tokens.append(ar_model.special_tokens["EOS"])  # End token

        # Show complete token sequence
        print("\n🔗 Complete Token Sequence:")
        print(f"   Length: {len(tokens)} tokens")
        print(f"   Sequence: {tokens}")

        # Decode token meanings
        print("\n📖 Token Sequence Explanation:")
        for i, token in enumerate(tokens):
            if token == ar_model.special_tokens["BOS"]:
                print(f"   Token {i}: {token} → BOS (Beginning of Sequence)")
            elif token == ar_model.special_tokens["EOS"]:
                print(f"   Token {i}: {token} → EOS (End of Sequence)")
            elif ar_model.is_gap_token(token):
                gap_length = ar_model.decode_gap_token(token)
                print(f"   Token {i}: {token} → GAP_{gap_length} ({gap_length} empty windows)")
            else:
                print(f"   Token {i}: {token} → HISTOGRAM_TOKEN (quantized medical data)")

        # Test autoregressive forward pass
        token_tensor = torch.tensor(tokens).unsqueeze(0)  # Add batch dim

        print("\n🚀 Autoregressive Forward Pass:")
        print(f"   Input shape: {token_tensor.shape}")

        with torch.no_grad():
            ar_result = ar_model(token_tensor)

        print(f"   AR Loss: {ar_result['loss'].item():.4f}")
        print(f"   Output logits shape: {ar_result['logits'].shape}")
        print("   → Can predict next token for each position!")


def demonstrate_generation_phase3(windows_by_patient: dict):
    """Demonstrate Phase 3: Generation/Inference (WITHOUT EOS token)."""

    print("\n🎯 PHASE 3: GENERATION/INFERENCE")
    print("=" * 60)
    print("Mode: GENERATION (Use partial sequence to generate continuation)")
    print("Key Difference: NO EOS token in input - model must generate it!")

    # Import trajectory generation functionality
    from morgen.generation.histogram_trajectories import format_histogram_trajectories

    # Create histogram model in GENERATION mode
    model_config = {
        "num_hidden_layers": 2,
        "num_attention_heads": 2,
        "hidden_size": 32,
        "max_position_embeddings": 50,
        "vocab_size": 50,  # Will be updated
    }
    quantizer_config = {"vocab_size": 8, "embedding_dim": 4, "n_embeddings": 16}
    gap_config = GapTokenConfig(max_gap_length=1200)

    # Create generation model
    gen_model = HistogramModel(
        model_config=model_config,
        quantizer_config=quantizer_config,
        gap_token_config=gap_config,
        mode=ModelMode.GENERATION,
    )

    print("\n📋 Generation Model Configuration:")
    print(f"   - Mode: {gen_model.mode}")
    print(f"   - Total vocabulary: {gen_model.total_vocab_size} tokens")
    print("   - Generation capability: Can continue sequences")

    # Create sequence processor for generation
    processor = HistogramSequenceProcessor(
        quantizer=gen_model.quantizer,
        vocab_size=quantizer_config["vocab_size"],
        max_gap_length=gap_config.max_gap_length,
        gap_token_strategy="multiple",
    )

    # Demonstrate generation for both patients
    for patient_id, windows in windows_by_patient.items():
        print(f"\n👤 PATIENT {patient_id} - Generation Demo:")
        print("-" * 50)

        # Create PARTIAL sequence for generation (key difference: NO EOS!)
        seed_tokens = [gen_model.special_tokens["BOS"]]  # Start with BOS only

        # Add first window as seed (if available)
        non_empty_windows = [w for w in windows if not w["is_gap"]]
        if len(non_empty_windows) > 0:
            # Use first histogram as seed
            first_histogram = non_empty_windows[0]["histogram"].unsqueeze(0)

            with torch.no_grad():
                z_e = gen_model.quantizer.encode(first_histogram)
                z_q, vq_loss, indices = gen_model.quantizer.quantize(z_e)
                first_token = indices[0].item()

            seed_tokens.append(first_token)

            # Show seed vs training difference
            print("🌱 Seed Sequence (for generation):")
            print("   Training:   [BOS, hist_token, ..., EOS] ← EOS provided")
            print("   Generation: [BOS, hist_token]           ← NO EOS, model generates!")
            print(f"   Actual seed: {seed_tokens}")

            # Mock generation (simplified version)
            print("\n🤖 Mock Generation Process:")
            current_sequence = seed_tokens.copy()
            max_new_tokens = 6

            for step in range(max_new_tokens):
                # Convert to tensor for model input
                input_tensor = torch.tensor(current_sequence).unsqueeze(0)

                with torch.no_grad():
                    # Get actual model predictions!
                    if gen_model.ar_model is not None:
                        # Use REAL autoregressive model for next token prediction
                        ar_result = gen_model(input_tensor)
                        logits = ar_result["logits"]  # Shape: [batch_size, seq_len, vocab_size]

                        # Get logits for the last position (next token prediction)
                        next_token_logits = logits[0, -1, :]  # Shape: [vocab_size]

                        # Sample from the distribution (or take argmax for deterministic)
                        # For demo clarity, let's use sampling with temperature
                        temperature = 1.0
                        probs = torch.softmax(next_token_logits / temperature, dim=0)
                        next_token = torch.multinomial(probs, 1).item()

                        # Show model's top predictions for educational purposes
                        top_k = 3
                        top_probs, top_indices = torch.topk(probs, top_k)
                        top_predictions = [
                            (idx.item(), prob.item()) for idx, prob in zip(top_indices, top_probs, strict=False)
                        ]

                        print(f"   Step {step + 1} - Model's top {top_k} predictions:")
                        for i, (token_idx, prob) in enumerate(top_predictions):
                            if token_idx == gen_model.special_tokens["EOS"]:
                                token_type = "EOS"
                            elif gen_model.is_gap_token(token_idx):
                                gap_len = gen_model.decode_gap_token(token_idx)
                                token_type = f"GAP_{gap_len}"
                            else:
                                token_type = "HIST_TOKEN"
                            print(f"      {i + 1}. Token {token_idx} ({token_type}) - prob: {prob:.3f}")

                        print(f"   → Model selected: Token {next_token}")

                    else:
                        # Fallback if AR model not available (shouldn't happen in GENERATION mode)
                        print("   ⚠️ AR model not available, using random token")
                        next_token = np.random.randint(0, gen_model.total_vocab_size)
                current_sequence.append(next_token)

                # Explain the token
                if next_token == gen_model.special_tokens["EOS"]:
                    print(f"   Step {step + 1}: Generated {next_token} → EOS (model ends sequence)")
                    break
                elif gen_model.is_gap_token(next_token):
                    gap_length = gen_model.decode_gap_token(next_token)
                    print(f"   Step {step + 1}: Generated {next_token} → GAP_{gap_length} (gap token)")
                else:
                    print(f"   Step {step + 1}: Generated {next_token} → HISTOGRAM_TOKEN")

            print("\n🔗 Complete Generated Sequence:")
            print(f"   Length: {len(current_sequence)} tokens")
            print(f"   Sequence: {current_sequence}")

            # Check if sequence ended naturally or hit limit
            if current_sequence[-1] == gen_model.special_tokens["EOS"]:
                print("   ✅ Sequence ended naturally with EOS token")
            else:
                print(f"   🚫 Sequence hit max_new_tokens limit ({max_new_tokens}) without EOS")
                print("   Note: In practice, you'd use longer max_length or early stopping")

            # Convert to trajectory format
            print("\n📊 Converting to MEDS Trajectory Format:")

            # Create code vocabulary for trajectory conversion
            code_vocab = {
                i: f"MED_CODE_{10 + i * 10:02d}" for i in range(8)
            }  # Maps 0->Code10, 1->Code20, etc.

            try:
                # Convert generated sequence to trajectories
                trajectories_df = format_histogram_trajectories(
                    generated_tokens=[torch.tensor(current_sequence)],
                    processor=processor,
                    code_vocab=code_vocab,
                    subject_ids=[patient_id + 1000],  # Use 1001, 1002
                    prediction_time=datetime(2020, 6, 1),
                    window_size_days=30,
                )

                if len(trajectories_df) > 0:
                    print(f"   ✅ Generated {len(trajectories_df)} trajectory observations")
                    print(f"   Columns: {list(trajectories_df.columns)}")

                    # Show sample trajectory data
                    print("\n📋 Sample Generated Observations:")
                    for i, row in enumerate(trajectories_df.iter_rows(named=True)):
                        if i >= 3:  # Show max 3 rows
                            print(f"      ... and {len(trajectories_df) - 3} more observations")
                            break
                        date_str = row["time"].strftime("%Y-%m-%d")
                        print(f"      {date_str}: {row['code']} (histogram_index={row['histogram_index']})")
                else:
                    print("   ⚠️  No observations generated (all histograms were empty)")

            except Exception as e:
                print(f"   ⚠️  Trajectory conversion failed: {e}")
        else:
            print("   ⚠️  No data available for seeding generation")

    print("\n💡 Phase 3 Key Insights:")
    print("   - Generation starts with [BOS] or [BOS, seed_token]")
    print("   - NO EOS in input - model must learn to generate it")
    print("   - Model controls sequence length via EOS generation")
    print("   - Output converts to MEDS-compatible trajectory format")
    print("   - Ready for medical timeline evaluation!")


def main():
    """Main demo function."""

    print("🎭 HISTOGRAM MODEL DEMONSTRATION")
    print("=" * 60)
    print("This demo shows the complete histogram model pipeline using")
    print("the same patient data from demo_histogram_dataset.py")
    print()

    # Create patient data
    patient_data = create_dummy_patient_data()

    # Create histogram windows
    print("\n📊 CREATING HISTOGRAM WINDOWS")
    print("=" * 60)
    print("Converting patient events into 30-day histogram windows...")

    windows_by_patient = create_mock_histogram_windows(patient_data, window_size_days=30)

    # Show window summaries
    for patient_id, windows in windows_by_patient.items():
        non_empty = sum(1 for w in windows if not w["is_gap"])
        empty = sum(1 for w in windows if w["is_gap"])
        print(f"\nPatient {patient_id}: {len(windows)} total windows ({non_empty} with data, {empty} empty)")

    # Phase 1: Vector Quantizer (ignore gaps)
    demonstrate_vector_quantizer_phase1(windows_by_patient)

    # Phase 2: Autoregressive Model (include gaps)
    demonstrate_autoregressive_phase2(windows_by_patient)

    # Phase 3: Generation/Inference (NO EOS in input)
    demonstrate_generation_phase3(windows_by_patient)

    print("\n🎉 DEMO COMPLETE!")
    print("=" * 60)
    print("Key Takeaways:")
    print("1. 🔧 Phase 1 (Quantizer): Compresses binary histograms → discrete tokens")
    print("2. 🤖 Phase 2 (Autoregressive): Learns sequences with gap tokens (WITH EOS)")
    print("3. 🎯 Phase 3 (Generation): Continues sequences without EOS → generates EOS")
    print("4. 📊 Patient 1: Regular timeline → few gaps")
    print("5. ⏳ Patient 2: Sparse timeline → long gap tokens (13 months!)")
    print("6. 📋 Complete pipeline: Train → Generate → MEDS trajectories")
    print("\nThe model can now generate realistic medical timelines for evaluation!")


if __name__ == "__main__":
    main()
