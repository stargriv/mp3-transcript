#!/usr/bin/env python3
"""
Voice Isolation and Transcription Script for TTS Learning

This script processes MP3 files to:
1. Isolate voice using vocal separation techniques
2. Generate transcripts with precise timestamps
3. Output data suitable for TTS model training

Dependencies: librosa, torch, pydub, openai-whisper, demucs
"""

import os
import sys
import json
import argparse
from pathlib import Path
from typing import List, Dict, Tuple
import warnings
warnings.filterwarnings("ignore")

try:
    import librosa
    import numpy as np
    from pydub import AudioSegment
    import torch
    import whisper
    import demucs.separate
    import soundfile as sf
except ImportError as e:
    print(f"Missing required library: {e}")
    print("Install with: pip install librosa torch pydub openai-whisper demucs soundfile")
    sys.exit(1)


class VoiceIsolator:
    """Handles voice isolation from audio files using Demucs."""
    
    def __init__(self, model_name: str = "htdemucs_ft", sample_rate: int = 44100):
        """
        Initialize VoiceIsolator with Demucs model.
        
        Args:
            model_name: Demucs model to use (htdemucs_ft, htdemucs, mdx_q, etc.)
            sample_rate: Target sample rate for output
        """
        self.sample_rate = sample_rate
        self.model_name = model_name
        print(f"🎯 Demucs model configured: {model_name}")
        
        # Store model configuration for demucs.separate usage
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"✅ Demucs model configured: {model_name} on {self.device}")
    
    def isolate_vocals(self, audio_path: str, output_path: str = None) -> str:
        """
        Isolate vocals using Demucs with simplified command-line interface.
        
        Args:
            audio_path: Path to input audio file
            output_path: Path for isolated vocal output
            
        Returns:
            Path to isolated vocal file
        """
        print(f"🎵 Processing audio with Demucs: {audio_path}")
        
        # Set up output path
        if output_path is None:
            base_name = Path(audio_path).stem
            output_path = f"{base_name}_vocals_isolated.wav"
        
        # Ensure output directory exists
        output_dir = Path(output_path).parent
        output_dir.mkdir(parents=True, exist_ok=True)
        
        try:
            print("🎯 Separating vocals with AI model...")
            print("⏳ Processing (this may take a few minutes)...")
            
            # Create temporary output directory for demucs
            temp_output_dir = output_dir / "temp_demucs"
            temp_output_dir.mkdir(exist_ok=True)
            
            # Use demucs.separate.main() with simplified parameters
            demucs_args = [
                "--two-stems", "vocals",  # Only separate vocals
                "-n", self.model_name,  # Use specified model
                "-o", str(temp_output_dir),  # Output directory
                audio_path
            ]
            
            # Run demucs separation
            demucs.separate.main(demucs_args)
            
            # Find the generated vocals file
            audio_basename = Path(audio_path).stem
            vocals_file = temp_output_dir / self.model_name / audio_basename / "vocals.wav"
            
            if not vocals_file.exists():
                raise FileNotFoundError(f"Demucs output not found at expected location: {vocals_file}")
            
            # Load the separated vocals and convert to desired format
            print(f"📁 Loading separated vocals: {vocals_file}")
            vocals_audio, _ = librosa.load(str(vocals_file), sr=self.sample_rate, mono=True)
            
            # Normalize
            vocals_audio = librosa.util.normalize(vocals_audio)
            
            # Save as WAV file
            print(f"💾 Saving isolated vocals: {output_path}")
            sf.write(output_path, vocals_audio, self.sample_rate)
            
            # Clean up temporary files
            import shutil
            shutil.rmtree(temp_output_dir)
            
            # Provide quality feedback
            vocal_power = np.mean(vocals_audio ** 2)
            print(f"✅ Vocal isolation complete! Quality metric: {vocal_power:.6f}")
            
            return output_path
                    
        except Exception as e:
            print(f"❌ Error during Demucs vocal separation: {e}")
            print("🔄 Falling back to traditional method...")
            
            # Clean up temp directory if it exists
            temp_output_dir = output_dir / "temp_demucs"
            if temp_output_dir.exists():
                import shutil
                shutil.rmtree(temp_output_dir)
            
            # Fallback to simple librosa method
            return self._fallback_isolation(audio_path, output_path)
    
    def _fallback_isolation(self, audio_path: str, output_path: str) -> str:
        """Fallback method using librosa if Demucs fails."""
        print("Using fallback librosa method...")
        
        # Load audio
        y, sr = librosa.load(audio_path, sr=self.sample_rate, mono=True)
        
        # Simple vocal isolation using harmonic-percussive separation
        y_harmonic, _ = librosa.effects.hpss(y, margin=(1.0, 5.0))
        
        # Normalize
        y_harmonic = librosa.util.normalize(y_harmonic)
        
        # Save
        sf.write(output_path, y_harmonic, sr)
        print(f"Fallback isolation saved: {output_path}")
        
        return output_path


class TranscriptGenerator:
    """Generates timestamped transcripts from audio."""
    
    def __init__(self, model_size: str = "tiny"):
        """
        Initialize with Whisper model.
        
        Args:
            model_size: Whisper model size (tiny, base, small, medium, large)
        """
        print(f"Loading Whisper model: {model_size}")
        self.model = whisper.load_model(model_size)
    
    def transcribe_with_timestamps(self, audio_path: str) -> Dict:
        """
        Generate transcript with word-level timestamps.
        
        Args:
            audio_path: Path to audio file
            
        Returns:
            Dictionary with transcript data
        """
        print(f"Transcribing: {audio_path}")
        
        # Transcribe with word timestamps
        result = self.model.transcribe(
            audio_path,
            word_timestamps=True,
            language="ru",  # Adjust based on your audio
            task="transcribe"
        )
        
        return result
    
    def format_for_tts(self, transcript_data: Dict, audio_duration: float = None) -> List[Dict]:
        """
        Format transcript data for TTS training.
        
        Args:
            transcript_data: Raw transcript from Whisper
            audio_duration: Total audio duration in seconds (optional)
            
        Returns:
            List of formatted segments for TTS training
        """
        tts_segments = []
        
        for segment in transcript_data.get("segments", []):
            segment_data = {
                "text": segment["text"].strip(),
                "start_time": round(segment["start"], 3),
                "end_time": round(segment["end"], 3),
                "duration": round(segment["end"] - segment["start"], 3),
                "confidence": round(segment.get("avg_logprob", 0), 3)
            }
            
            # Add word-level timestamps if available
            if "words" in segment:
                segment_data["words"] = [
                    {
                        "word": word["word"].strip(),
                        "start": round(word["start"], 3),
                        "end": round(word["end"], 3),
                        "probability": round(word.get("probability", 0), 3)
                    }
                    for word in segment["words"]
                ]
            
            tts_segments.append(segment_data)
        
        return tts_segments


def main():
    parser = argparse.ArgumentParser(
        description="Isolate voice and generate timestamped transcript for TTS learning"
    )
    parser.add_argument("input_file", help="Input MP3 file path")
    parser.add_argument("--output-dir", default="output", 
                       help="Output directory (default: output)")
    parser.add_argument("--whisper-model", default="tiny",
                       choices=["tiny", "base", "small", "medium", "large"],
                       help="Whisper model size (default: tiny)")
    parser.add_argument("--remove-vocals", action="store_true",
                       help="Remove isolated vocal file after processing (default: keep)")
    parser.add_argument("--force-isolation", action="store_true",
                       help="Force re-extraction of voice even if file exists")
    parser.add_argument("--force-transcription", action="store_true",
                       help="Force re-transcription even if file exists")
    parser.add_argument("--demucs-model", default="htdemucs_ft",
                       choices=["htdemucs_ft", "htdemucs", "mdx_q", "mdx", "mdx_extra_q"],
                       help="Demucs model for voice separation (default: htdemucs_ft)")
    
    args = parser.parse_args()
    
    # Validate input file
    if not os.path.exists(args.input_file):
        print(f"Error: Input file not found: {args.input_file}")
        sys.exit(1)
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)
    
    input_path = Path(args.input_file)
    base_name = input_path.stem
    
    # Step 1: Isolate vocals
    print("\n=== STEP 1: Voice Isolation ===")
    vocals_path = output_dir / f"{base_name}_vocals.wav"
    
    # Check if voice isolation already exists
    if vocals_path.exists() and not args.force_isolation:
        print(f"Voice isolation already exists: {vocals_path}")
        print("Skipping voice isolation (use --force-isolation to re-extract)")
        isolated_vocals = str(vocals_path)
    else:
        if vocals_path.exists():
            print(f"Re-extracting voice (--force-isolation specified)")
        isolator = VoiceIsolator(model_name=args.demucs_model)
        isolated_vocals = isolator.isolate_vocals(args.input_file, str(vocals_path))
    
    # Step 2: Generate transcript with timestamps
    print("\n=== STEP 2: Transcription ===")
    transcript_file = output_dir / f"{base_name}_transcript.json"
    tts_file = output_dir / f"{base_name}_tts_data.json"
    
    # Check if transcription already exists
    if transcript_file.exists() and tts_file.exists() and not args.force_transcription:
        print(f"Transcription already exists: {transcript_file}")
        print("Skipping transcription (use --force-transcription to re-transcribe)")
        
        # Load existing data for summary
        with open(transcript_file, "r", encoding="utf-8") as f:
            transcript_data = json.load(f)
        with open(tts_file, "r", encoding="utf-8") as f:
            tts_output = json.load(f)
        
        tts_segments = tts_output.get("segments", [])
        audio_duration = tts_output.get("metadata", {}).get("total_duration", 0)
        
    else:
        if transcript_file.exists() or tts_file.exists():
            print(f"Re-transcribing (--force-transcription specified)")
        
        transcriber = TranscriptGenerator(args.whisper_model)
        transcript_data = transcriber.transcribe_with_timestamps(isolated_vocals)
        
        # Get audio duration for validation
        audio = AudioSegment.from_file(isolated_vocals)
        audio_duration = len(audio) / 1000.0  # Convert to seconds
        
        # Step 3: Format for TTS training
        print("\n=== STEP 3: Formatting for TTS ===")
        tts_segments = transcriber.format_for_tts(transcript_data, audio_duration)
        
        # Full transcript data
        with open(transcript_file, "w", encoding="utf-8") as f:
            json.dump(transcript_data, f, ensure_ascii=False, indent=2)
        
        # TTS training data
        tts_output = {
            "metadata": {
                "original_file": args.input_file,
                "vocals_file": isolated_vocals,
                "total_duration": audio_duration,
                "whisper_model": args.whisper_model,
                "language": transcript_data.get("language", "unknown"),
                "segments_count": len(tts_segments)
            },
            "segments": tts_segments
        }
        
        with open(tts_file, "w", encoding="utf-8") as f:
            json.dump(tts_output, f, ensure_ascii=False, indent=2)
    
    # Generate summary
    total_text_duration = sum(seg["duration"] for seg in tts_segments)
    coverage = (total_text_duration / audio_duration) * 100 if audio_duration > 0 else 0
    
    print(f"\n=== RESULTS ===")
    print(f"Original file: {args.input_file}")
    print(f"Isolated vocals: {isolated_vocals}")
    print(f"Full transcript: {transcript_file}")
    print(f"TTS training data: {tts_file}")
    print(f"\nStats:")
    print(f"  Total audio duration: {audio_duration:.2f}s")
    print(f"  Speech coverage: {coverage:.1f}%")
    print(f"  Segments: {len(tts_segments)}")
    print(f"  Language detected: {transcript_data.get('language', 'unknown')}")
    
    # Cleanup isolated vocals only if explicitly requested
    if args.remove_vocals:
        os.remove(isolated_vocals)
        print(f"  Removed isolated vocals file as requested")
    else:
        print(f"  Isolated vocals preserved: {isolated_vocals}")
    
    print(f"\nTTS training data ready at: {tts_file}")


if __name__ == "__main__":
    main()