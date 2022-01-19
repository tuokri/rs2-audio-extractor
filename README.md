# rs2-audio-extractor

Decode Wwise audio files to Ogg format
while preserving original file names and project structure
where possible.

Works with Rising Storm 2: Vietnam & Killing Floor 2. May also work with other games that provide Wwise metadata (txt) files.

Good performance by taking advantage of multiple CPU cores
by using multiprocessing.

## Usage

Show help:

`extract.exe --help`

Parse files in `WwiseAudio` and write output in `output` in Ogg format:

`extract.exe "C:\...\WwiseAudio" "C:\output"`

Number of CPU cores to use may be set with the `--max-workers X` argument,
where X is the number of CPU cores to use. All available CPU cores 
are used by default.

## Download

From releases: https://github.com/tuokri/rs2-audio-extractor/releases

## Troubleshooting 

### ww2ogg errors or errors opening decoded files

The program output directory will have paths longer than the max default Windows path limit. 

Fix by enabling Win32 long paths and rebooting.

https://www.howtogeek.com/266621/how-to-make-windows-10-accept-file-paths-over-260-characters/
