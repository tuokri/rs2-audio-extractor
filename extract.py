import argparse
import os
import re
import shutil
import subprocess
import sys
import time
from concurrent import futures
from concurrent.futures.process import ProcessPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import List
from typing import Tuple

from logbook import FileHandler
from logbook import Logger
from logbook import StreamHandler

_sh = StreamHandler(sys.stdout, level="INFO", bubble=True)
_fh = FileHandler("extract.log", mode="w", level="INFO", bubble=True)
logger = Logger(__name__)
logger.handlers.append(_sh)
logger.handlers.append(_fh)


def do_log_stuff(subscriber):
    with subscriber:
        subscriber.dispatch_forever()


if getattr(sys, "frozen", False):
    # noinspection PyUnresolvedReferences,PyProtectedMember
    BASE_PATH = Path(sys._MEIPASS)
else:
    BASE_PATH = Path("")

MAX_PATH = 260
WAVESCAN = BASE_PATH / Path("bin/wavescan.bms")
QUICKBMS = BASE_PATH / Path("bin/quickbms.exe")
WW2OGG = BASE_PATH / Path("bin/ww2ogg.exe")
PCB = BASE_PATH / Path("bin/packed_codebooks_aoTuV_603.bin")
MAX_WORKERS = 1

MEMORY_BANK_PAT = re.compile(
    r"^\t+([0-9]+)\t+([\-\w ]+)\t+([\-\w:\\.() ]+)\t+(\\[\w \-\\]+)\t+([0-9]+)$"
)
STREAMED_BANK_PAT = re.compile(
    r"^\t+([0-9]+)\t+([\-\w ]+)\t+([\-\w:\\.() ]+)\t+([\-\w:\\.() ]+)\t+(\\[\w \-\\]+)\t+$"
)
QUICKBMS_OUT_PAT = re.compile(r"\s{2}[0-9]+\s([0-9]+)\s+(.*)\r\n")


@dataclass
class BankMetaData:
    audio_id: int
    name: str
    audio_source_file: str
    wwise_object_path: str = ""
    generated_audio_file: str = ""
    notes: str = ""
    data_size: int = 0

    def is_streamed_audio(self) -> bool:
        return True if self.generated_audio_file else False


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
    ap.add_argument(
        "--workers",
        default=os.cpu_count(),
        type=int,
        help="number of processors to user, (default=%(default)s)",
    )
    return ap.parse_args()


def parse_banks_metadata(wwise_dir: Path) -> Tuple[dict, dict]:
    fut2txt_file = {}
    memory_bnk_file2metadata = {}
    streamed_bnk_file2metadata = {}
    with ProcessPoolExecutor(max_workers=MAX_WORKERS) as executor:
        for txt_file in wwise_dir.rglob("*.txt"):
            fut2txt_file[executor.submit(parse_bank_metadata, txt_file)] = txt_file

        for completed_fut in futures.as_completed(fut2txt_file):
            meta_result = completed_fut.result()
            txt_file = fut2txt_file[completed_fut]
            memory_bnk_file2metadata[txt_file.stem] = meta_result[0]
            # No stem here!
            streamed_bnk_file2metadata[txt_file] = meta_result[1]

    return memory_bnk_file2metadata, streamed_bnk_file2metadata


def parse_memory_audio_meta(line: str):
    match = re.match(MEMORY_BANK_PAT, line)
    return BankMetaData(
        audio_id=int(match.group(1)),
        name=match.group(2),
        audio_source_file=match.group(3),
        wwise_object_path=match.group(4),
        # notes=match.group(5),
        data_size=int(match.group(5)),
    )


def parse_streamed_audio_meta(line: str):
    match = re.match(STREAMED_BANK_PAT, line)
    return BankMetaData(
        audio_id=int(match.group(1)),
        name=match.group(2),
        audio_source_file=match.group(3),
        generated_audio_file=match.group(4),
        wwise_object_path=match.group(5),
        # notes=match.group(5),
    )


def parse_bank_metadata(bank: Path) -> Tuple[List[BankMetaData], List[BankMetaData]]:
    streamed_flag = False
    memory_flag = False
    ret_memory = []
    ret_streamed = []
    try:
        with bank.open("r", encoding="utf-8") as f:
            for line in f:
                if line.lower().startswith("in memory audio"):
                    memory_flag = True
                    continue
                elif line.lower().startswith("streamed audio"):
                    streamed_flag = True
                    continue
                elif line == "\r\n" or line == "\n":
                    streamed_flag = False
                    memory_flag = False
                    continue

                if memory_flag:
                    ret_memory.append(parse_memory_audio_meta(line))
                elif streamed_flag:
                    ret_streamed.append(parse_streamed_audio_meta(line))

    except Exception as e:
        logger.error("error parsing {bank}: {e}", bank=bank, e=repr(e))
    return ret_memory, ret_streamed


def decode_banks(wwise_dir: Path, out_dir: Path) -> dict:
    orig_bnk2decode_info = {}
    fut2orig_bnk = {}
    with ProcessPoolExecutor(max_workers=MAX_WORKERS) as executor:
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
        logger.info("decoding '{b}'...", b=bnk_file.absolute())
        quickbms_out = subprocess.check_output(
            [str(QUICKBMS.absolute()), "-o", str(WAVESCAN.absolute()),
             str(bnk_file.absolute()), str(out_dir.absolute())],
            stderr=subprocess.STDOUT,
        )
        try:
            quickbms_out = quickbms_out.decode("utf-8")
        except Exception as e:
            logger.info("error decoding QuickBMS output for "
                        "'{bnk_file}': {e}", bnk_file=bnk_file, e=repr(e))
    except subprocess.CalledProcessError as cpe:
        logger.error("error processing '{bnk_file}': {cpe}",
                     bnk_file=bnk_file, cpe=repr(cpe), exc_info=True)

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
        src.replace(dst)
        logger.info("moved '{src}' -> '{dst}'",
                    src=src.absolute(), dst=dst.absolute())
    except Exception as e:
        logger.error(
            "error moving: '{src}' to '{dst}': {e}",
            src=src.absolute(),
            dst=dst.absolute(),
            e=e,
            exc_info=True
        )


def copy(src: Path, dst: Path):
    try:
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(src, dst)
        logger.info("copied '{src}' -> '{dst}'",
                    src=src.absolute(), dst=dst.absolute())
    except Exception as e:
        logger.error(
            "error copying: '{src}' to '{dst}': {e}",
            src=src.absolute(),
            dst=dst.absolute(),
            e=e,
            exc_info=True
        )


def ww2ogg(src: Path):
    out = b""
    try:
        logger.info("converting '{src}' to OGG", src=src.absolute())
        out = subprocess.check_output(
            [WW2OGG, str(src.absolute()), "--pcb", str(PCB.absolute())],
            stderr=subprocess.STDOUT)
        src.unlink()
        logger.info("removed {src}", src=src.absolute())
    except subprocess.CalledProcessError as cpe:
        logger.error(
            "ww2ogg error for '{src}': code={code}, out={o}",
            src=src,
            code=cpe.returncode,
            o=out.decode("utf-8"),
            exc_info=True,
        )


def main():
    start = time.time()

    global MAX_WORKERS

    args = parse_args()
    MAX_WORKERS = args.workers
    wwise_dir = Path(args.wwise_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(exist_ok=True)

    logger.info("processing audio files in '{wd}'", wd=wwise_dir.absolute())

    fut2func = {}
    # Parse .bnk files and metadata.
    with ProcessPoolExecutor(max_workers=MAX_WORKERS) as executor:
        fut2func[executor.submit(parse_banks_metadata, wwise_dir)] = parse_banks_metadata
        fut2func[executor.submit(decode_banks, wwise_dir, out_dir)] = decode_banks

    memory_bnk_meta_file2metadata = {}
    streamed_bnk_meta_file2metadata = {}
    orig_bnk2decode_info = {}
    for completed_fut in futures.as_completed(fut2func):
        if fut2func[completed_fut] == parse_banks_metadata:
            result = completed_fut.result()
            memory_bnk_meta_file2metadata = result[0]
            streamed_bnk_meta_file2metadata = result[1]
        elif fut2func[completed_fut] == decode_banks:
            orig_bnk2decode_info = completed_fut.result()

    if len(memory_bnk_meta_file2metadata) != len(orig_bnk2decode_info):
        raise ValueError(
            f"Amount of Bank and metadata files "
            f"do not match ({len(orig_bnk2decode_info)} "
            f"!= {len(memory_bnk_meta_file2metadata)})")

    wem_src2wem_dst = {}
    # Move .wem files to out_dir in correct places.
    with ProcessPoolExecutor(max_workers=MAX_WORKERS) as executor:
        for bnk_meta_file, meta in streamed_bnk_meta_file2metadata.items():
            for m in meta:
                src_dir = bnk_meta_file.parent
                src = src_dir / Path(m.generated_audio_file)
                if src.exists():
                    wwise_path = Path(m.wwise_object_path)
                    dst = out_dir / wwise_path.relative_to(
                        wwise_path.anchor).with_suffix(".wem")
                    executor.submit(copy, src, dst)
                    wem_src2wem_dst[src] = dst
                else:
                    logger.warning(
                        "found references to {src} in metadata, but "
                        "the file cannot be found in wwise_dir", src=src)

    decoded_file2metas = {}

    for orig_bnk_file, decode_info in orig_bnk2decode_info.items():
        orig_bnk_file = orig_bnk_file.stem
        meta = memory_bnk_meta_file2metadata[orig_bnk_file]

        if len(decode_info) != len(meta):
            raise ValueError(f"decode_info and meta length mismatch "
                             f"{len(decode_info)} != {len(meta)}")

        for m, (decoded_stem, decoded_size) in zip(meta, decode_info.items()):
            if m.data_size != decoded_size:
                raise ValueError(f"{m.data_size} != {decoded_size}")
            decoded_file2metas[decoded_stem] = m

    fs = []
    # Move output from decoding .bnk files to correct places in out_dir.
    executor = ProcessPoolExecutor(max_workers=MAX_WORKERS)
    for decoded_file, meta in decoded_file2metas.items():
        src = out_dir / f"{decoded_file}.bin"
        wwise_path = Path(meta.wwise_object_path)
        dst = out_dir / wwise_path.relative_to(wwise_path.anchor).with_suffix(".bin")
        fs.append(executor.submit(move, src, dst))

    futures.wait(fs, return_when=futures.ALL_COMPLETED)

    fs = []
    # Convert all .wem and .bin files to .ogg.
    executor = ProcessPoolExecutor(max_workers=MAX_WORKERS)
    for bin_file in out_dir.rglob("*.bin"):
        fs.append(executor.submit(ww2ogg, bin_file))
    for wem_file in out_dir.rglob("*.wem"):
        fs.append(executor.submit(ww2ogg, wem_file))

    futures.wait(fs, return_when=futures.ALL_COMPLETED)

    done_wems_stems = set([ws.stem for ws in wem_src2wem_dst.keys()])
    source_wems = [w for w in wwise_dir.rglob("*.wem")]
    source_wems_stems = set([w.stem for w in source_wems])
    wem_diff = source_wems_stems.difference(done_wems_stems)
    if wem_diff:
        logger.warn("failed to determine filename for "
                    "the the following {num} files:", num=len(wem_diff))
    for ws in source_wems:
        if ws.stem in wem_diff:
            logger.info(ws)
            copy(ws, out_dir)

    # Convert leftovers.
    with ProcessPoolExecutor(max_workers=MAX_WORKERS) as executor:
        for wem_file in out_dir.rglob("*.wem"):
            fs.append(executor.submit(ww2ogg, wem_file))

    secs = time.time() - start
    logger.info("finished successfully in {secs:.2f} seconds", secs=secs)


if __name__ == "__main__":
    import multiprocessing

    multiprocessing.freeze_support()

    main()
