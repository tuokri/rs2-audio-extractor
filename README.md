# rs2-audio-extractor

Decode Wwise audio files to Ogg format
while preserving original file names and project structure
where possible.

Works with Rising Storm 2: Vietnam & Killing Floor 2. May also work with other games that provide Wwise metadata (txt) files.

Good performance by taking advantage of multiple CPU cores
by using multiprocessing.

## Usage

**Show help:**

`extract.exe --help`

**Parse files in `WwiseAudio` and write output in `output` in Ogg format:**

`extract.exe "C:\...\WwiseAudio" "C:\output"`

## Concrete usage example commands

**First open PowerShell or cmd.exe in the folder where you extracted extract.exe**

**Extract KF2 Russian audio:**

`.\extract.exe J:\SteamLibrary\steamapps\common\killingfloor2\KFGame\BrewedPC\WwiseAudio\Russian\ H:\KF2Audio_Russian`

**Extract KF2 English audio:**

`.\extract.exe J:\SteamLibrary\steamapps\common\killingfloor2\KFGame\BrewedPC\WwiseAudio\Windows H:\KF2Audio_English`

**Extract RS2 audio:**

`.\extract.exe "C:\Program Files (x86)\Steam\steamapps\common\Rising Storm 2\ROGame\BrewedPC\WwiseAudio\Windows" H:\RS2_Audio`

## Download

From releases: https://github.com/tuokri/rs2-audio-extractor/releases

## Troubleshooting 

### ww2ogg errors or errors opening decoded files

The program output directory will have paths longer than the max default Windows path limit. 

Fix by enabling Win32 long paths and rebooting.

https://www.howtogeek.com/266621/how-to-make-windows-10-accept-file-paths-over-260-characters/

## Technical details

Third party software used:

QuickBMS: https://aluigi.altervista.org/quickbms.htm

The extractor also uses my custom forks of the following software:

ww2ogg: https://github.com/tuokri/ww2ogg

revorbstd: https://github.com/tuokri/revorbstd
