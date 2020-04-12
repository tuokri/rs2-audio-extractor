# rs2-audio-extractor

Decode Rising Storm 2: Vietnam Wwise audio files
while preserving original file names and project structure
where possible.

## Usage

Show help:

`extract.exe --help`

Parse files in `Wwiseaudio` and write output in `output`:

`extract.exe "C:\...\WwiseAudio" "C:\output"`

## Download

From releases: https://github.com/tuokri/rs2-audio-extractor/releases

## Troubleshooting 

### ww2ogg errors

The program output directory will have paths
longer than the max default Windows path limit. 

Fix by enabling Win32 long paths and rebooting.

https://www.howtogeek.com/266621/how-to-make-windows-10-accept-file-paths-over-260-characters/

### Problematic file warning

Some audio source files have multiple possible final output name
candidates so they will have to be handle manually.

Problematic files are place in the root of the output dir.
