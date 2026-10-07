"""Whisper large-v3 turbo in plain torch: a second hearing of a song's words, for YuE2 Transcribe.

``decode`` is the long-form transcription's bookkeeping and has no
dependencies. ``network``, ``model`` and ``runtime`` need torch and are
imported only when a song is actually heard. The weights are OpenAI's own
``openai/whisper-large-v3-turbo`` release, MIT.
"""
