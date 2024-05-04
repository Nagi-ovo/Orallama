# Orallama

Great thanks to the reference project of this work: [apeatling/ollama-voice-mac](https://github.com/apeatling/ollama-voice-mac)

## Installing and running🦙

1. Install [Ollama](https://ollama.ai).
2. Download the a model, Mistral 7b is a actually a ood choice using the `ollama pull mistral` command.
3. Download an [OpenAI Whisper Model](https://github.com/openai/whisper/discussions/63#discussioncomment-3798552) and place the `.pt` model file in a `/whisper` directory in the repo root folder.
4. Make sure you have [FFmpeg](https://ffmpeg.org/) by running `brew install ffmpeg`.
5. For Apple silicon support of the PyAudio library you'll need to install [Homebrew](https://brew.sh) and run `brew install portaudio`.
6. Run `poetry install` to install all dependencies (`pip install -r requirements.txt` also works).
7. Run `python chat.py` to start the assistant.
