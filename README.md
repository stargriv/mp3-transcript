# MP3 Transcript

A Python-based audio processing tool for TTS (Text-to-Speech) learning. This tool processes MP3 files to isolate voice content and generate timestamped transcripts suitable for training TTS models.

## Features

- **AI-Powered Voice Isolation**: Uses Facebook Research's Demucs model for superior vocal separation
- **Timestamped Transcription**: Generates word-level timestamps using OpenAI Whisper
- **Multiple Output Formats**: Creates both full transcription and TTS-ready formatted data
- **Intelligent Processing**: Skips re-processing existing files unless forced
- **Fallback Support**: Automatically falls back to librosa if Demucs fails

## Setup

1. Clone the repository:
```bash
git clone https://github.com/stargriv/mp3-transcript.git
cd mp3-transcript
```

2. Create and activate virtual environment:
```bash
python3 -m venv venv
source venv/bin/activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

Basic usage:
```bash
python voice_isolate_transcript.py "input/audio_file.mp3"
```

Advanced options:
```bash
# Use larger Whisper model for better accuracy
python voice_isolate_transcript.py "input/audio_file.mp3" --whisper-model base

# Custom output directory
python voice_isolate_transcript.py "input/audio_file.mp3" --output-dir custom_output

# Use different Demucs model
python voice_isolate_transcript.py "input/audio_file.mp3" --demucs-model htdemucs

# Remove vocal file after processing
python voice_isolate_transcript.py "input/audio_file.mp3" --remove-vocals

# Force re-processing
python voice_isolate_transcript.py "input/audio_file.mp3" --force-isolation --force-transcription
```

## Output Files

The script generates these files in the output directory:
- `{filename}_transcript.json`: Full Whisper transcription data
- `{filename}_tts_data.json`: Formatted segments for TTS training
- `{filename}_vocals.wav`: Isolated vocal audio file

## Dependencies

- **demucs**: AI model for music source separation
- **openai-whisper**: Speech-to-text transcription
- **librosa**: Audio processing utilities
- **torch**: Backend for AI models
- **pydub**: Audio file format handling
- **soundfile**: Audio file I/O
- **numpy**: Numerical computing

## Directory Structure

```
mp3-transcript/
├── input/              # Place MP3 files here
├── output/             # Generated files (auto-created)
├── voice_isolate_transcript.py  # Main script
├── requirements.txt    # Dependencies
└── README.md          # This file
```

## Notes

- Configured for Russian language transcription by default
- GPU acceleration supported for faster processing
- Available Demucs models: htdemucs_ft (default), htdemucs, mdx_q, mdx, mdx_extra_q