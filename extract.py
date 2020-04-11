import argparse
import re
import subprocess
from concurrent import futures
from concurrent.futures.process import ProcessPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import List

WAVESCAN = Path("wavescan.bms")
QUICKBMS = Path("quickbms.exe")

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
    bank_data = {}
    with ProcessPoolExecutor() as executor:
        for txt_file in wwise_dir.rglob("*.txt"):
            fut2txt_file[executor.submit(parse_bank_metadata, txt_file)] = txt_file

        for completed_fut in futures.as_completed(fut2txt_file):
            result = completed_fut.result()
            if result:
                txt_file = fut2txt_file[completed_fut]
                bank_data[txt_file.stem] = result

    return bank_data


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
    ret = {}
    futs = []
    with ProcessPoolExecutor() as executor:
        for bnk_file in wwise_dir.rglob("*.bnk"):
            futs.append(executor.submit(decode_bank, bnk_file, out_dir))

    for completed_fut in futures.as_completed(futs):
        for k, v in completed_fut.result().items():
            ret[k] = v

    return ret


def decode_bank(bnk_file: Path, out_dir: Path) -> dict:
    quickbms_out = []
    try:
        print(f"decoding '{bnk_file.absolute()}'...")
        quickbms_out = subprocess.check_output(
            [str(QUICKBMS.absolute()), "-o", str(WAVESCAN.absolute()),
             str(bnk_file.absolute()), str(out_dir.absolute())],
            stderr=subprocess.STDOUT
        )
        try:
            quickbms_out = quickbms_out.decode("utf-8")
        except Exception as e:
            print(f"error decoding QuickBMS output for '{bnk_file}': {repr(e)}")
    except subprocess.CalledProcessError as cpe:
        print(f"error processing '{bnk_file}': {cpe}")

    filename2datasize = {}
    matches = re.finditer(QUICKBMS_OUT_PAT, quickbms_out)
    for match in matches:
        dsize = match.group(1)
        fname = match.group(2)
        filename2datasize[Path(fname).stem] = dsize

    return filename2datasize


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

    metas = {}
    fname2dsize = {}
    for completed_fut in futures.as_completed(fut2func):
        if fut2func[completed_fut] == parse_banks_metadata:
            metas = completed_fut.result()
        elif fut2func[completed_fut] == decode_banks:
            fname2dsize = completed_fut.result()

    for bnk_meta_file, metadata in metas.items():
        print(bnk_meta_file)

    for fname, dsize in fname2dsize.items():
        original_bnk_stem = "_".join(Path(fname).stem.split("_")[:-1])


if __name__ == '__main__':
    main()
