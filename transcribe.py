# Command line command example:
#   whisper "C:\Timur\home\english\youtube-playlist-downloader\bebebears\Bebebears - Episode 1 - Star story - Super ToonsTV.mp4" --language en --output_format srt                         whisper "C:\Timur\home\english\youtube-playlist-downloader\bebebears\Bebebears - Episode 1 - Star story - Super ToonsTV.mp4" --language en --output_format srt
# Stdout output:
#   [00:00.000 --> 00:09.000]  Taking walks and playing games when there's sunny weather
#   [00:09.000 --> 00:15.000]  If there's rain, they're singing songs or reading books together
#   ...
# Srt output:
#   1
#   00:00:00,000 --> 00:00:09,000
#   Taking walks and playing games when there's sunny weather
#
#   2
#   00:00:09,000 --> 00:00:15,000
#   If there's rain, they're singing songs or reading books together
#   ...

# Python example (not tested):
# from pathlib import Path
# import whisper
#
# model = whisper.load_model('large')
#
# for file in Path(r'path\to\your\folder').glob('*.mp4'):
#     result = model.transcribe(str(file))
#
#     with open(file.with_suffix('.txt'), 'w') as f:
#         f.write(result['text'])