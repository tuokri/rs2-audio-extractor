import argparse
import multiprocessing as mp
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

import logbook
from logbook.queues import MultiProcessingHandler
from logbook.queues import MultiProcessingSubscriber

# if getattr(sys, "frozen", False):
#     # noinspection PyUnresolvedReferences,PyProtectedMember
#     BASE_PATH = Path(sys._MEIPASS)
# else:
#     BASE_PATH = Path("")
BASE_PATH = Path("")

REVORB = BASE_PATH / Path("bin/RevorbStd.exe")
WAVESCAN = BASE_PATH / Path("bin/wavescan.bms")
QUICKBMS = BASE_PATH / Path("bin/quickbms.exe")
WW2OGG = BASE_PATH / Path("bin/ww2ogg.exe")
PCB = BASE_PATH / Path("bin/packed_codebooks_aoTuV_603.bin")
MAX_WORKERS = os.cpu_count()  # TODO: --workers argument doesn't work for now.
ID_TO_FILENAME = "id_to_filename.txt"
SENTINEL = (None, None)

MEMORY_BANK_PAT = re.compile(
    r"^\t+([0-9]+)\t+([\-\w ,()_]+)\t+([_\-\w:\\.() ,]+)\t+(\\[_()\w \-\\,]+)\t+([0-9]+)$"
)
# STREAMED_BANK_PAT = re.compile(
#     r"^\t+([0-9]+)\t+([\-\w ]+)\t+([\-\w:\\.() ]+)\t+([\-\w:\\.() ]+)\t+(\\[\w \-\\]+)\t+$"
# )
STREAMED_BANK_PAT = re.compile(
    r"^\t+([0-9]+)\t+([\-\w ]+)\t+([\-\w:\\.() ]+)\t+([\-\w:\\.() ]+)\t+(\\[\w \-\\]+)\t*(.*)\t*$"
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


def setup_logging(queue: mp.Queue):
    logbook.set_datetime_format("local")
    handler = MultiProcessingHandler(queue)
    handler.push_application()


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


def parse_banks_metadata(wwise_dir: Path, queue: mp.Queue) -> Tuple[dict, dict]:
    fut2txt_file = {}
    memory_bnk_file2metadata = {}
    streamed_bnk_file2metadata = {}
    with ProcessPoolExecutor(max_workers=MAX_WORKERS) as executor:
        for txt_file in wwise_dir.rglob("*.txt"):
            fut2txt_file[executor.submit(parse_bank_metadata, txt_file, queue)] = txt_file

        for completed_fut in futures.as_completed(fut2txt_file):
            meta_result = completed_fut.result()
            txt_file = fut2txt_file[completed_fut]
            memory_bnk_file2metadata[txt_file.stem] = meta_result[0]
            # No stem here!
            streamed_bnk_file2metadata[txt_file] = meta_result[1]

    return memory_bnk_file2metadata, streamed_bnk_file2metadata


def parse_memory_audio_meta(line: str):
    match = re.match(MEMORY_BANK_PAT, line)
    if not match:
        raise ValueError(f"can't match MEMORY_BANK_PAT for line: '{line}'")
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


def parse_bank_metadata(bank: Path, queue: mp.Queue) -> Tuple[List[BankMetaData], List[BankMetaData]]:
    setup_logging(queue)

    streamed_flag = False
    memory_flag = False
    ret_memory = []
    ret_streamed = []
    try:
        logbook.info("parsing bank metadata '{bank}'", bank=bank.absolute())
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
        logbook.exception("error parsing {bank}: {e}", bank=bank, e=repr(e))
    return ret_memory, ret_streamed


def decode_banks(wwise_dir: Path, out_dir: Path, quickbms_log,
                 quickbms_log_lock, queue: mp.Queue) -> dict:
    setup_logging(queue)

    orig_bnk2decode_info = {}
    fut2orig_bnk = {}
    with ProcessPoolExecutor(max_workers=MAX_WORKERS) as executor:
        for bnk_file in wwise_dir.rglob("*.bnk"):
            fut2orig_bnk[executor.submit(decode_bank, bnk_file, out_dir,
                                         quickbms_log, quickbms_log_lock, queue)] = bnk_file

    for completed_fut in futures.as_completed(fut2orig_bnk):
        result = completed_fut.result()
        orig_bnk = fut2orig_bnk[completed_fut]
        orig_bnk2decode_info[orig_bnk] = result

    return orig_bnk2decode_info


def decode_bank(bnk_file: Path, out_dir: Path, quickbms_log: Path,
                quickbms_log_lock: mp.Lock, queue: mp.Queue) -> dict:
    setup_logging(queue)

    quickbms_out = ""
    try:
        logbook.info("decoding '{b}'...", b=bnk_file.absolute())
        quickbms_out = subprocess.check_output(
            [str(QUICKBMS.absolute()), "-o", str(WAVESCAN.absolute()),
             str(bnk_file.absolute()), str(out_dir.absolute())],
            stderr=subprocess.STDOUT,
        )
        with quickbms_log_lock:
            with quickbms_log.open("ab") as f:
                f.write(b"*" * 10)
                f.write(b" ")
                f.write(bytes(bnk_file.absolute()))
                f.write(b" ")
                f.write(b"*" * 10)
                f.write(b"\n")
                f.write(quickbms_out)
                f.write(b"\n")
        try:
            quickbms_out = quickbms_out.decode("utf-8")
        except Exception as e:
            logbook.info("error decoding QuickBMS output for "
                         "'{bnk_file}': {e}", bnk_file=bnk_file, e=repr(e))
    except subprocess.CalledProcessError as cpe:
        logbook.error("error processing '{bnk_file}': {cpe}",
                      bnk_file=bnk_file, cpe=repr(cpe), exc_info=True)

    filename2datasize = {}
    matches = re.finditer(QUICKBMS_OUT_PAT, quickbms_out)
    for match in matches:
        dsize = int(match.group(1))
        fname = match.group(2)
        filename2datasize[Path(fname).stem] = dsize

    return filename2datasize


def move(src: Path, dst: Path, queue: mp.Queue, id_to_filename_queue: mp.Queue = None):
    setup_logging(queue)

    try:
        dst.parent.mkdir(parents=True, exist_ok=True)
        src.replace(dst)
        if id_to_filename_queue:
            id_to_filename_queue.put((src, dst))
        logbook.info("moved '{src}' -> '{dst}'",
                     src=src.absolute(), dst=dst.absolute())
    except Exception as e:
        logbook.error(
            "error moving: '{src}' to '{dst}': {e}",
            src=src.absolute(),
            dst=dst.absolute(),
            e=e,
            exc_info=True
        )


# No parallel logging setup here.
def copy_seq(src: Path, dst: Path, id_to_filename_queue: mp.Queue = None):
    try:
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(src, dst)
        if id_to_filename_queue:
            id_to_filename_queue.put((src, dst))
        logbook.info("copied '{src}' -> '{dst}'",
                     src=src.absolute(), dst=dst.absolute())
    except Exception as e:
        logbook.error(
            "error copying: '{src}' to '{dst}': {e}",
            src=src.absolute(),
            dst=dst.absolute(),
            e=e,
            exc_info=True
        )


def copy(src: Path, dst: Path, queue: mp.Queue,
         id_to_filename_queue: mp.Queue = None):
    setup_logging(queue)
    copy_seq(src, dst, id_to_filename_queue)


def revorb(ogg_file: Path, queue: mp.Queue):
    setup_logging(queue)

    logbook.info("running revorb on {f}", f=ogg_file.absolute())
    try:
        subprocess.check_call([REVORB, str(ogg_file.absolute())])
    except subprocess.CalledProcessError as cpe:
        logbook.exception("error running revorb on {f}: {e}",
                          f=ogg_file.absolute(), e=cpe)


def ww2ogg(src: Path, queue: mp.Queue):
    setup_logging(queue)

    out = b""
    try:
        logbook.info("converting '{src}' to OGG", src=src.resolve().absolute())
        out = subprocess.check_output(
            [WW2OGG, str(src.resolve().absolute()), "--pcb", str(PCB.resolve().absolute())],
            stderr=subprocess.STDOUT)
        src.unlink()
        logbook.info("removed {src}", src=src.resolve().absolute())
        revorb(src.with_suffix(".ogg"), queue)
    except subprocess.CalledProcessError as cpe:
        logbook.exception(
            "ww2ogg error for '{src}': code={code}, out={o}",
            src=src.resolve().absolute(),
            code=cpe.returncode,
            o=out.decode("utf-8"),
        )


def id_to_filename_worker(id_queue: mp.Queue, out_file: Path, queue: mp.Queue):
    setup_logging(queue)

    f = None
    try:
        f = out_file.open(mode="a")
        item = id_queue.get()
        while item != SENTINEL:
            try:
                f.write(f"{item[0].name}\t{item[1].name}\n")
            except Exception as e:
                logbook.error(e, exc_info=True)
            item = id_queue.get()
        logbook.info("id_to_filename_worker got sentinel, stopping")
    except Exception as e:
        logbook.error(e, exc_info=True)
    finally:
        if f:
            f.close()


def main():
    start = time.time()

    args = parse_args()
    wwise_dir = Path(args.wwise_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(exist_ok=True)
    id_to_filename_path = out_dir / ID_TO_FILENAME

    manager = mp.Manager()
    queue = manager.Queue()
    id_queue = manager.Queue()

    setup_logging(queue)

    target_handlers = logbook.NestedSetup([
        logbook.NullHandler(),
        logbook.StreamHandler(sys.stdout, level=logbook.INFO, bubble=True),
        logbook.FileHandler("extract.log", mode="w", level=logbook.INFO, bubble=True),
    ])

    sub = MultiProcessingSubscriber(queue)
    controller = sub.dispatch_in_background(target_handlers)

    quickbms_log_lock = manager.Lock()
    quickbms_log = out_dir / "quickbms.log"

    try:
        id_to_filename_path.unlink()
        logbook.info("removed old {id_file}", id_file=id_to_filename_path.absolute())
    except FileNotFoundError:
        pass

    logbook.info("QuickBMS log: '{qlog}'", qlog=quickbms_log.absolute())
    try:
        quickbms_log.unlink()
        logbook.info("removed old {f}", f=quickbms_log.absolute())
    except FileNotFoundError:
        pass

    id_to_filename_path.touch()
    logbook.info("writing old ID -> new filename info in '{id_file}'",
                 id_file=id_to_filename_path.absolute())

    id_to_filename_p = mp.Process(target=id_to_filename_worker,
                                  args=(id_queue, id_to_filename_path, queue))
    id_to_filename_p.start()

    logbook.info("processing audio files in '{wd}'", wd=wwise_dir.absolute())

    fut2func = {}
    # Parse .bnk files and metadata.
    with ProcessPoolExecutor(max_workers=MAX_WORKERS) as executor:
        fut2func[executor.submit(parse_banks_metadata, wwise_dir, queue)] = parse_banks_metadata
        fut2func[executor.submit(
            decode_banks, wwise_dir,
            out_dir, quickbms_log, quickbms_log_lock, queue)] = decode_banks

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
        logbook.warning(
            "Amount of Bank and metadata files "
            "do not match ({first}) != {second})",
            first=len(orig_bnk2decode_info),
            second=len(memory_bnk_meta_file2metadata)
        )

        s1 = memory_bnk_meta_file2metadata.keys()
        s2 = set([key.stem for key in orig_bnk2decode_info])

        to_del = []
        diff = s2.difference(s1)
        for d in diff:
            # TODO: expensive!
            for key in orig_bnk2decode_info:
                if key.stem == d:
                    logbook.warn("ignoring {f}", f=str(key))
                    to_del.append(key)

        for td in to_del:
            del orig_bnk2decode_info[td]

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
                    executor.submit(copy, src, dst, queue, id_queue)
                    wem_src2wem_dst[src] = dst
                else:
                    logbook.warning(
                        "found references to {src} in metadata, but "
                        "the file cannot be found in wwise_dir", src=src)

    decoded_file2metas = {}

    for orig_bnk_file, decode_info in orig_bnk2decode_info.items():
        orig_bnk_file = orig_bnk_file.stem
        meta = memory_bnk_meta_file2metadata[orig_bnk_file]

        if len(decode_info) != len(meta):
            logbook.error("decode_info and meta length mismatch: "
                          "{len1} != {len2} for bnk: '{bnk}'",
                          len1=len(decode_info), len2=len(meta),
                          bnk=orig_bnk_file)
            # print(decode_info)
            # print(meta)
            continue
            # raise ValueError(f"decode_info and meta length mismatch "
            #                  f"{len(decode_info)} != {len(meta)}")

        for m, (decoded_stem, decoded_size) in zip(meta, decode_info.items()):
            if m.data_size != decoded_size:
                # raise ValueError(f"{m.data_size} != {decoded_size}")
                logbook.error("metadata size and decoded data size length mismatch: "
                              "{len1} != {len2}",
                              len1=m.data_size, len2=decoded_size)
                continue
            decoded_file2metas[decoded_stem] = m

    fs = []
    # Move output from decoding .bnk files to correct places in out_dir.
    executor = ProcessPoolExecutor(max_workers=MAX_WORKERS)
    for decoded_file, meta in decoded_file2metas.items():
        src = out_dir / f"{decoded_file}.bin"
        wwise_path = Path(meta.wwise_object_path)
        dst = out_dir / wwise_path.relative_to(wwise_path.anchor).with_suffix(".bin")
        fs.append(executor.submit(move, src, dst, queue, id_queue))

    futures.wait(fs, return_when=futures.ALL_COMPLETED)

    fs = []
    # Convert all .wem and .bin files to .ogg.
    executor = ProcessPoolExecutor(max_workers=MAX_WORKERS)
    for bin_file in out_dir.rglob("*.bin"):
        fs.append(executor.submit(ww2ogg, bin_file, queue))
    for wem_file in out_dir.rglob("*.wem"):
        fs.append(executor.submit(ww2ogg, wem_file, queue))

    futures.wait(fs, return_when=futures.ALL_COMPLETED)

    done_wems_stems = set([ws.stem for ws in wem_src2wem_dst.keys()])
    source_wems = [w for w in wwise_dir.rglob("*.wem")]
    source_wems_stems = set([w.stem for w in source_wems])
    wem_diff = source_wems_stems.difference(done_wems_stems)

    if wem_diff:
        logbook.warn("failed to determine filename for "
                     "{num} files", num=len(wem_diff))

    for ws in source_wems:
        if str(ws.stem) in wem_diff:
            logbook.info("couldn't determine filename for: {ws}", ws=ws)
            copy_seq(ws, out_dir, id_to_filename_queue=None)

    # Convert leftovers.
    leftovers_fs = []
    with ProcessPoolExecutor(max_workers=MAX_WORKERS) as executor:
        for wem_file in out_dir.rglob("*.wem"):
            leftovers_fs.append(executor.submit(ww2ogg, wem_file, queue))

    futures.wait(leftovers_fs, return_when=futures.ALL_COMPLETED)

    id_queue.put(SENTINEL)
    id_to_filename_p.join()

    secs = time.time() - start
    logbook.info("finished successfully in {secs:.2f} seconds", secs=secs)

    controller.stop()


if __name__ == "__main__":
    import multiprocessing

    multiprocessing.freeze_support()

    main()
