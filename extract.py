import argparse
import re
import subprocess
from concurrent import futures
from concurrent.futures.process import ProcessPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import List

WAVESCAN = Path("bin/wavescan.bms")
QUICKBMS = Path("bin/quickbms.exe")
WW2OGG = Path("bin/ww2ogg.exe")
PCB = Path("bin/packed_codebooks_aoTuV_603.bin")

BANK_PAT = re.compile(r"^\t+([0-9]+)\t+(\w+)\t+([\w:\\.]+)\t+(\\[\w \-\\]+)\t+([0-9]+)$")
QUICKBMS_OUT_PAT = re.compile(r"\s{2}[0-9]+\s([0-9]+)\s+(.*)\r\n")


@dataclass
class BankMetaData:
    in_memory_audio_id: int
    name: str
    wwise_object_path: str
    notes: str
    data_size: int


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "wwise_dir",
        help="path to directory containing Wwise audio files"
    )
    ap.add_argument(
        "out_dir",
        help="path to output directory",
    )
    return ap.parse_args()


def parse_banks_metadata(wwise_dir: Path) -> dict:
    fut2txt_file = {}
    bnk_file2metadata = {}
    with ProcessPoolExecutor() as executor:
        for txt_file in wwise_dir.rglob("*.txt"):
            fut2txt_file[executor.submit(parse_bank_metadata, txt_file)] = txt_file

        for completed_fut in futures.as_completed(fut2txt_file):
            result = completed_fut.result()
            txt_file = fut2txt_file[completed_fut]
            bnk_file2metadata[txt_file.stem] = result

    return bnk_file2metadata


def parse_bank_metadata(bank: Path) -> List[BankMetaData]:
    ret = []
    try:
        with bank.open("r") as f:
            for line in f:
                match = re.match(BANK_PAT, line)
                if match:
                    ret.append(BankMetaData(
                        in_memory_audio_id=int(match.group(1)),
                        name=match.group(2),
                        wwise_object_path=match.group(3),
                        notes=match.group(4),
                        data_size=int(match.group(5)),
                    ))
    except Exception as e:
        print(f"error parsing {bank.absolute()}: {repr(e)}")
    return ret


def decode_banks(wwise_dir: Path, out_dir: Path) -> dict:
    orig_bnk2decode_info = {}
    fut2orig_bnk = {}
    with ProcessPoolExecutor() as executor:
        for bnk_file in wwise_dir.rglob("*.bnk"):
            fut2orig_bnk[executor.submit(decode_bank, bnk_file, out_dir)] = bnk_file

    for completed_fut in futures.as_completed(fut2orig_bnk):
        result = completed_fut.result()
        orig_bnk = fut2orig_bnk[completed_fut]
        orig_bnk2decode_info[orig_bnk] = result

    return orig_bnk2decode_info


def decode_bank(bnk_file: Path, out_dir: Path) -> dict:
    quickbms_out = []
    try:
        print(f"decoding '{bnk_file.absolute()}'...")
        quickbms_out = subprocess.check_output(
            [str(QUICKBMS.absolute()), "-o", str(WAVESCAN.absolute()),
             str(bnk_file.absolute()), str(out_dir.absolute())],
            stderr=subprocess.STDOUT,
        )
        try:
            quickbms_out = quickbms_out.decode("utf-8")
        except Exception as e:
            print(f"error decoding QuickBMS output for '{bnk_file}': {repr(e)}")
    except subprocess.CalledProcessError as cpe:
        print(f"error processing '{bnk_file}': {repr(cpe)}")

    filename2datasize = {}
    matches = re.finditer(QUICKBMS_OUT_PAT, quickbms_out)
    for match in matches:
        dsize = int(match.group(1))
        fname = match.group(2)
        filename2datasize[Path(fname).stem] = dsize

    return filename2datasize


def move(src: Path, dst: Path):
    try:
        dst.parent.mkdir(parents=True, exist_ok=True)
        src.rename(dst)
        print(f"moved '{src.absolute()}' -> '{dst.absolute()}'")
    except Exception as e:
        print(f"error moving: '{src.absolute()}' to '{dst.absolute()}'"
              f": {repr(e)}")


def ww2ogg(src: Path):
    out = b""
    try:
        print(f"converting '{src.absolute()}' to OGG")
        out = subprocess.check_output(
            [WW2OGG, str(src.absolute()), "--pcb", str(PCB.absolute())],
            stderr=subprocess.STDOUT)
        src.unlink()
        print(f"removed {src.absolute()}")
    except subprocess.CalledProcessError as cpe:
        print(out.decode("utf-8"))
        print(f"ww2ogg error for '{src}': {repr(cpe)}: "
              f"code={cpe.returncode}")


def main():
    args = parse_args()
    wwise_dir = Path(args.wwise_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(exist_ok=True)

    print(f"processing audio files in '{wwise_dir.absolute()}'")

    fut2func = {}
    with ProcessPoolExecutor() as executor:
        fut2func[(executor.submit(parse_banks_metadata, wwise_dir))] = parse_banks_metadata
        fut2func[(executor.submit(decode_banks, wwise_dir, out_dir))] = decode_banks

    bnk_meta_file2metadata = {}
    orig_bnk2decode_info = {}
    for completed_fut in futures.as_completed(fut2func):
        if fut2func[completed_fut] == parse_banks_metadata:
            bnk_meta_file2metadata = completed_fut.result()
        elif fut2func[completed_fut] == decode_banks:
            orig_bnk2decode_info = completed_fut.result()

    if len(bnk_meta_file2metadata) != len(orig_bnk2decode_info):
        raise ValueError(
            f"Amount of Bank and metadata files "
            f"do not match ({len(orig_bnk2decode_info)} != {len(bnk_meta_file2metadata)})")

    decoded_file2metas = {}

    # Let's do it the lazy fucking way and go for the triple-nested for-loop.
    for orig_bnk_file, decode_info in orig_bnk2decode_info.items():
        orig_bnk_file = orig_bnk_file.stem
        meta = bnk_meta_file2metadata[orig_bnk_file]
        for m in meta:
            for decoded_file, decoded_data_size in decode_info.items():
                if decoded_data_size == m.data_size:
                    if decoded_file in decoded_file2metas:
                        decoded_file2metas[decoded_file] = [decoded_file2metas[decoded_file]]
                        decoded_file2metas[decoded_file].append(m)
                    else:
                        decoded_file2metas[decoded_file] = m

    fs = []
    problematic = []
    with ProcessPoolExecutor() as executor:
        for decoded_file, meta in decoded_file2metas.items():
            if isinstance(meta, List):
                problematic.append((decoded_file, meta))
                continue

            src = out_dir / f"{decoded_file}.bin"
            notes_path = Path(meta.notes)
            dst = out_dir / notes_path.relative_to(notes_path.anchor).with_suffix(".bin")
            fs.append(executor.submit(move, src, dst))

    futures.wait(fs, return_when=futures.ALL_COMPLETED)

    fs = []
    with ProcessPoolExecutor() as executor:
        for bin_file in out_dir.rglob("*.bin"):
            fs.append(executor.submit(ww2ogg, bin_file))

    futures.wait(fs, return_when=futures.ALL_COMPLETED)

    if problematic:
        print("******************* WARNING *******************")
        print("******************* WARNING *******************")
        print("******************* WARNING *******************")

    for prob in problematic:
        print(f"problematic file '{Path(prob[0]).absolute()}' with multiple "
              f"possible output options skipped, please handle it manually")

    if problematic:
        print("***********************************************")
        print("***********************************************")
        print("***********************************************")


if __name__ == "__main__":
    main()