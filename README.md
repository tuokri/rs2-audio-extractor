# rs2-audio-extractor

Decode Rising Storm 2: Vietnam Wwise audio files
while preserving original file names and project structure
where possible.

## Usage

Show help:

`extract.exe --help`

Parse files in `Wwiseaudio` and write output in `output` in Ogg format:

`extract.exe "C:\...\WwiseAudio" "C:\output"`

## Download

From releases: https://github.com/tuokri/rs2-audio-extractor/releases

## Troubleshooting 

### ww2ogg errors or errors opening decoded files

The program output directory will have paths
longer than the max default Windows path limit. 

Fix by enabling Win32 long paths and rebooting.

https://www.howtogeek.com/266621/how-to-make-windows-10-accept-file-paths-over-260-characters/
