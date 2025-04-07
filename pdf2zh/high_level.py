"""Functions that can be used for the most common use-cases for pdf2zh.six"""

import asyncio
import io
import logging
import multiprocessing as mp
import os
import re
import sys
import tempfile
import time
from asyncio import CancelledError
from multiprocessing.pool import AsyncResult
from pathlib import Path
from string import Template
from typing import Any, BinaryIO, Literal

import numpy as np
import requests
import tqdm
import tqdm.contrib.concurrent
from babeldoc.assets.assets import get_font_and_metadata
from pdfminer.converter import PDFPageAggregator
from pdfminer.layout import LAParams, LTTextBox, LTTextLine
from pdfminer.pdfdocument import PDFDocument
from pdfminer.pdfexceptions import PDFValueError
from pdfminer.pdfinterp import PDFPageInterpreter, PDFResourceManager
from pdfminer.pdfpage import PDFPage
from pdfminer.pdfparser import PDFParser
from pymupdf import Document, Font, Point, Rect
from pymupdf import Page as MupdfPage

from pdf2zh.config import ConfigManager
from pdf2zh.converter import TranslateConverter
from pdf2zh.doclayout import ModelInstance
from pdf2zh.pdfinterp import PDFPageInterpreterEx

NOTO_NAME = "noto"

logger = logging.getLogger(__name__)

noto_list = [
    "am",  # Amharic
    "ar",  # Arabic
    "bn",  # Bengali
    "bg",  # Bulgarian
    "chr",  # Cherokee
    "el",  # Greek
    "gu",  # Gujarati
    "iw",  # Hebrew
    "hi",  # Hindi
    "kn",  # Kannada
    "ml",  # Malayalam
    "mr",  # Marathi
    "ru",  # Russian
    "sr",  # Serbian
    "ta",  # Tamil
    "te",  # Telugu
    "th",  # Thai
    "ur",  # Urdu
    "uk",  # Ukrainian
]

colors = (
    np.array(
        [
            [102, 102, 255],  # 0:title, rgb(102, 102, 255)
            [153, 0, 76],  # 1:plain text, rgb(153, 0, 76)
            [158, 158, 158],  # 2:abandon, rgb(158, 158, 158)
            [153, 255, 51],  # 3:figure, rgb(153, 255, 51)
            [102, 178, 255],  # 4:figure caption, rgb(102, 178, 255)
            [204, 204, 0],  # 5:table, rgb(204, 204, 0)
            [255, 255, 102],  # 6:table caption, rgb(255, 255, 102)
            [229, 255, 204],  # 7:table footnote, rgb(229, 255, 204)
            [0, 255, 0],  # 8:isolate formula, rgb(0, 255, 0)
            [40, 169, 92],  # 9:formula caption, rgb(40, 169, 92)
        ]
    )
    / 255.0
)


def check_files(files: list[str]) -> list[str]:
    files = [f for f in files if not f.startswith("http://")]  # exclude online files, http
    files = [f for f in files if not f.startswith("https://")]  # exclude online files, https
    missing_files = [file for file in files if not os.path.exists(file)]
    return missing_files


def doclayout_patch(
    inf: BinaryIO,
    doc_zh: Document,
    pages: list[int],
    cancellation_event: asyncio.Event = None,
    **kwarg: Any,
):
    doc_debug = None
    if logger.isEnabledFor(logging.DEBUG):
        doc_debug = Document(stream=doc_zh.tobytes())
    model = ModelInstance.value

    layout = {}

    parser = PDFParser(inf)
    doc = PDFDocument(parser)

    pdf_pages: list[PDFPage] = []
    boxes: list[np.ndarray] = []
    images: list[np.ndarray] = []
    for pageno, page in enumerate(PDFPage.create_pages(doc)):
        if cancellation_event and cancellation_event.is_set():
            raise CancelledError("task cancelled")
        if pages and (pageno not in pages):
            continue
        page.pageno = pageno
        pdf_pages.append(page)
        pix = doc_zh[page.pageno].get_pixmap()
        image = np.frombuffer(pix.samples, np.uint8).reshape(pix.height, pix.width, 3)[:, :, ::-1]
        images.append(image)
        boxes.append(np.ones((pix.height, pix.width)))

    layout_detection_start = time.time()
    yolo_results = model.predict(images, batch_size=1)
    layout_detection_end = time.time()
    logger.info(
        f"Layout Detection Time ({len(yolo_results)} pages): {(layout_detection_end - layout_detection_start):.4f}"
    )
    # yolo_results = sum([model.predict(im, imgsz=int(im.shape[0] / 32) * 32) for im in images], [])

    # 按 block 矫正左右边界
    rsrcmgr = PDFResourceManager()
    device = PDFPageAggregator(rsrcmgr, laparams=LAParams())
    interpreter = PDFPageInterpreter(rsrcmgr, device)
    calibrate_bbox_start = time.time()
    for box, page, yolo_res in zip(boxes, pdf_pages, yolo_results, strict=True):
        h, w = box.shape
        interpreter.process_page(page)
        text_boxes = [e for e in device.get_result() if isinstance(e, LTTextBox)]
        for _, d in enumerate(yolo_res.boxes):
            x0, y0, x1, y1 = d.xyxy.squeeze()
            for tb in text_boxes:
                line = next(filter(lambda e: isinstance(e, LTTextLine), tb), None)
                if line is None:
                    continue
                lh = line.height
                tbx0, tby0, tbx1, tby1 = tb.bbox
                tbx0, tby0, tbx1, tby1 = (
                    np.clip(tbx0 - 1, 0, w - 1),
                    np.clip(h - tby1 - 1, 0, h - 1),
                    np.clip(tbx1 + 1, 0, w - 1),
                    np.clip(h - tby0 + 1, 0, h - 1),
                )
                w_tolerance = 0.2 * tb.width
                if (
                    y0 <= tby0 + lh / 2 and y1 >= tby1 - lh / 2  # block 上下边界在检测框之内 (放宽半个行高)，并且：
                ) and (
                    tbx0 <= x0 <= tbx0 + w_tolerance  # block 左边界比检测框小 20%
                    or tbx1 - w_tolerance <= x1 <= tbx1  # block 右边界比检测框宽大 20%
                ):
                    d.xyxy[0] = min(x0, tbx0)
                    d.xyxy[2] = max(x1, tbx1)
    calibrate_bbox_end = time.time()
    logger.info(f"Calibrate Detection Bbox Time: {(calibrate_bbox_end - calibrate_bbox_start):.4f}")

    for box, page, yolo_res in zip(boxes, pdf_pages, yolo_results, strict=True):
        if cancellation_event and cancellation_event.is_set():
            raise CancelledError("task cancelled")
        h, w = box.shape
        vcls = ["abandon", "figure", "table", "isolate_formula", "formula_caption"]
        for i, d in enumerate(yolo_res.boxes):
            if doc_debug:
                c, conf = int(d.cls), float(d.conf)
                name = model._names[c]
                color = colors[c].tolist()
                label_text = f"{name}: {conf:.2f}"
                x0, y0, x1, y1 = d.xyxy.squeeze()
                mupdf_page: MupdfPage = doc_debug[page.pageno]
                tm = mupdf_page.derotation_matrix
                mupdf_page.draw_rect(
                    Rect(x0, y0, x1, y1) * tm,
                    color=None,
                    fill=color,
                    fill_opacity=0.3,
                    width=0.5,
                    overlay=True,
                )
                mupdf_page.insert_text(
                    Point(x1 + 2, y0 + 10) * tm,
                    label_text,
                    fontsize=10,
                    color=color,
                    rotate=mupdf_page.rotation,
                )
            if yolo_res.names[int(d.cls)] not in vcls:
                x0, y0, x1, y1 = d.xyxy.squeeze()
                x0, y0, x1, y1 = (
                    np.clip(int(x0 - 1), 0, w - 1),
                    np.clip(int(h - y1 - 1), 0, h - 1),
                    np.clip(int(x1 + 1), 0, w - 1),
                    np.clip(int(h - y0 + 1), 0, h - 1),
                )
                box[y0:y1, x0:x1] = i + 2
        for _, d in enumerate(yolo_res.boxes):
            if yolo_res.names[int(d.cls)] in vcls:
                x0, y0, x1, y1 = d.xyxy.squeeze()
                x0, y0, x1, y1 = (
                    np.clip(int(x0 - 1), 0, w - 1),
                    np.clip(int(h - y1 - 1), 0, h - 1),
                    np.clip(int(x1 + 1), 0, w - 1),
                    np.clip(int(h - y0 + 1), 0, h - 1),
                )
                box[y0:y1, x0:x1] = 0
        layout[page.pageno] = box
        # 新建一个 xref 存放新指令流
        page.page_xref = doc_zh.get_new_xref()  # hack 插入页面的新 xref
        doc_zh.update_object(page.page_xref, "<<>>")
        doc_zh.update_stream(page.page_xref, b"")
        doc_zh[page.pageno].set_contents(page.page_xref)

    return layout, pdf_pages, doc_debug


def translate_patch(
    fp: BinaryIO,
    pageno: int,
    xref: int,
    box: np.ndarray,
    vfont: str = "",
    vchar: str = "",
    thread: int = 0,
    lang_in: str = "",
    lang_out: str = "",
    service: str = "",
    noto_name: str = "",
    font_path: str = "",
    cancellation_event: asyncio.Event = None,
    envs: dict = None,
    prompt: Template = None,
    ignore_cache: bool = False,
    max_retries: int = 10,
    error: Literal["raise", "source", "drop"] = "source",
    **kwarg: Any,
) -> dict:
    iter_pages = PDFPage.create_pages(PDFDocument(PDFParser(fp)))
    current_idx = 0
    pdf_page = None
    while current_idx < pageno:
        pdf_page = next(iter_pages)
        current_idx += 1
    pdf_page = next(iter_pages)
    pdf_page.page_xref = xref
    pdf_page.pageno = pageno

    rsrcmgr = PDFResourceManager()
    device = TranslateConverter(
        rsrcmgr,
        vfont,
        vchar,
        thread,
        box,
        lang_in,
        lang_out,
        service,
        noto_name,
        font_path,
        envs,
        prompt,
        ignore_cache,
        max_retries=max_retries,
        error=error,
    )

    assert device is not None
    obj_patch = {}
    interpreter = PDFPageInterpreterEx(rsrcmgr, device, obj_patch)

    interpreter.process_page(pdf_page)

    device.close()
    return obj_patch


def translate_stream(
    stream: bytes,
    pages: list[int] | None = None,
    lang_in: str = "",
    lang_out: str = "",
    service: str = "",
    thread: int = 0,
    workers: int = 1,
    vfont: str = "",
    vchar: str = "",
    callback: object = None,
    cancellation_event: asyncio.Event = None,
    envs: dict = None,
    prompt: Template = None,
    skip_subset_fonts: bool = False,
    ignore_cache: bool = False,
    max_retries: int = 10,
    error: Literal["raise", "source", "drop"] = "source",
    onnx: str = None,
    **kwarg: Any,
):
    font_list = [("tiro", None)]

    font_path = download_remote_fonts(lang_out.lower())
    noto_name = NOTO_NAME
    noto = Font(noto_name, font_path)
    font_list.append((noto_name, font_path))

    doc_en = Document(stream=stream)
    stream = io.BytesIO()
    doc_en.save(stream)
    doc_zh = Document(stream=stream)
    page_count = doc_zh.page_count
    # font_list = [("GoNotoKurrent-Regular.ttf", font_path), ("tiro", None)]
    font_id = {}
    for page in doc_zh:
        for font in font_list:
            font_id[font[0]] = page.insert_font(font[0], font[1])
    xreflen = doc_zh.xref_length()
    for xref in range(1, xreflen):
        for label in ["Resources/", ""]:  # 可能是基于 xobj 的 res
            try:  # xref 读写可能出错
                font_res = doc_zh.xref_get_key(xref, f"{label}Font")
                target_key_prefix = f"{label}Font/"
                if font_res[0] == "xref":
                    resource_xref_id = re.search("(\\d+) 0 R", font_res[1]).group(1)
                    xref = int(resource_xref_id)
                    font_res = ("dict", doc_zh.xref_object(xref))
                    target_key_prefix = ""

                if font_res[0] == "dict":
                    for font in font_list:
                        target_key = f"{target_key_prefix}{font[0]}"
                        font_exist = doc_zh.xref_get_key(xref, target_key)
                        if font_exist[0] == "null":
                            doc_zh.xref_set_key(
                                xref,
                                target_key,
                                f"{font_id[font[0]]} 0 R",
                            )
            except Exception:
                pass

    fp = io.BytesIO()

    doc_zh.save(fp)

    if pages:
        total_pages = len(pages)
    else:
        total_pages = doc_zh.page_count
        pages = list(range(total_pages))

    layout, pdf_pages, doc_debug = doclayout_patch(
        fp,
        doc_zh,
        pages,
        cancellation_event,
        onnx,
    )

    obj_patches = []

    with tqdm.tqdm(total=total_pages) as progress:

        def cb(*args, **kwargs):
            progress.update(1)
            if callback:
                callback(progress)

        with mp.Pool(workers) as pool:
            async_results: list[AsyncResult] = []
            for arg in [
                (
                    fp,
                    p.pageno,
                    p.page_xref,
                    layout[p.pageno],
                    vfont,
                    vchar,
                    thread,
                    lang_in,
                    lang_out,
                    service,
                    noto_name,
                    font_path,
                    cancellation_event,
                    envs,
                    prompt,
                    ignore_cache,
                    max_retries,
                    error,
                )
                for p in pdf_pages
            ]:
                async_results.append(pool.apply_async(translate_patch, args=arg, callback=cb))
            obj_patches = [res.get() for res in async_results]

    for obj_patch in obj_patches:
        for obj_id, ops_new in obj_patch.items():
            # ops_old=doc_en.xref_stream(obj_id)
            # print(obj_id)
            # print(ops_old)
            # print(ops_new.encode())
            doc_zh.update_stream(obj_id, ops_new.encode())

    doc_en.insert_file(doc_zh)
    for id in range(page_count):
        doc_en.move_page(page_count + id, id * 2 + 1)

    if not skip_subset_fonts:
        try:
            try:
                doc_zh_copy = Document(stream=doc_zh.write())
                doc_zh_copy.subset_fonts()
                doc_en_copy = Document(stream=doc_en.write())
                doc_en_copy.subset_fonts()
                doc_zh = doc_zh_copy
                doc_en = doc_en_copy
            except Exception:
                logger.warning("Trying the fallback method to subset fonts", stack_info=True)
                doc_zh_copy = Document(stream=doc_zh.write())
                doc_zh_copy.subset_fonts(fallback=True)
                doc_en_copy = Document(stream=doc_en.write())
                doc_en_copy.subset_fonts(fallback=True)
                doc_zh = doc_zh_copy
                doc_en = doc_en_copy
        except Exception:
            logger.warning("Failed to subset fonts, skip.", stack_info=True)
    return (
        doc_zh.write(deflate=True, garbage=3, use_objstms=1),
        doc_en.write(deflate=True, garbage=3, use_objstms=1),
        doc_debug.write(deflate=True, garbage=3, use_objstms=1) if doc_debug else None,
    )


def convert_to_pdfa(input_path, output_path):
    """
    Convert PDF to PDF/A format

    Args:
        input_path: Path to source PDF file
        output_path: Path to save PDF/A file
    """
    from pikepdf import Dictionary, Name, Pdf

    # Open the PDF file
    pdf = Pdf.open(input_path)

    # Add PDF/A conformance metadata
    metadata = {
        "pdfa_part": "2",
        "pdfa_conformance": "B",
        "title": pdf.docinfo.get("/Title", ""),
        "author": pdf.docinfo.get("/Author", ""),
        "creator": "PDF Math Translate",
    }

    with pdf.open_metadata() as meta:
        meta.load_from_docinfo(pdf.docinfo)
        meta["pdfaid:part"] = metadata["pdfa_part"]
        meta["pdfaid:conformance"] = metadata["pdfa_conformance"]

    # Create OutputIntent dictionary
    output_intent = Dictionary(
        {
            "/Type": Name("/OutputIntent"),
            "/S": Name("/GTS_PDFA1"),
            "/OutputConditionIdentifier": "sRGB IEC61966-2.1",
            "/RegistryName": "http://www.color.org",
            "/Info": "sRGB IEC61966-2.1",
        }
    )

    # Add output intent to PDF root
    if "/OutputIntents" not in pdf.Root:
        pdf.Root.OutputIntents = [output_intent]
    else:
        pdf.Root.OutputIntents.append(output_intent)

    # Save as PDF/A
    pdf.save(output_path, linearize=True)
    pdf.close()


def translate(
    files: list[str],
    output: str = "",
    pages: list[int] | None = None,
    lang_in: str = "",
    lang_out: str = "",
    service: str = "",
    thread: int = 0,
    workers: int = 1,
    vfont: str = "",
    vchar: str = "",
    callback: object = None,
    compatible: bool = False,
    cancellation_event: asyncio.Event = None,
    envs: dict = None,
    prompt: Template = None,
    skip_subset_fonts: bool = False,
    ignore_cache: bool = False,
    max_retries: int = 10,
    error: Literal["raise", "source", "drop"] = "source",
    onnx: str = None,
    **kwarg: Any,
):
    if not files:
        raise PDFValueError("No files to process.")

    missing_files = check_files(files)

    if missing_files:
        print("The following files do not exist:", file=sys.stderr)
        for file in missing_files:
            print(f"  {file}", file=sys.stderr)
        raise PDFValueError("Some files do not exist.")

    result_files = []

    for file in files:
        if type(file) is str and (file.startswith("http://") or file.startswith("https://")):
            print("Online files detected, downloading...")
            try:
                r = requests.get(file, allow_redirects=True)
                if r.status_code == 200:
                    with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp_file:
                        print(f"Writing the file: {file}...")
                        tmp_file.write(r.content)
                        file = tmp_file.name
                else:
                    r.raise_for_status()
            except Exception as e:
                raise PDFValueError(
                    f"Errors occur in downloading the PDF file. Please check the link(s).\nError:\n{e}"
                )
        filename = os.path.splitext(os.path.basename(file))[0]

        # If the commandline has specified converting to PDF/A format
        # --compatible / -cp
        if compatible:
            with tempfile.NamedTemporaryFile(suffix="-pdfa.pdf", delete=False) as tmp_pdfa:
                print(f"Converting {file} to PDF/A format...")
                convert_to_pdfa(file, tmp_pdfa.name)
                doc_raw = open(tmp_pdfa.name, "rb")
                os.unlink(tmp_pdfa.name)
        else:
            doc_raw = open(file, "rb")
        s_raw = doc_raw.read()
        doc_raw.close()

        temp_dir = Path(tempfile.gettempdir())
        file_path = Path(file)
        try:
            if file_path.exists() and file_path.resolve().is_relative_to(temp_dir.resolve()):
                file_path.unlink(missing_ok=True)
                logger.debug(f"Cleaned temp file: {file_path}")
        except Exception:
            logger.warning(f"Failed to clean temp file {file_path}", exc_info=True)

        s_mono, s_dual, s_debug = translate_stream(
            s_raw,
            **locals(),
        )
        file_mono = Path(output) / f"{filename}-mono.pdf"
        file_dual = Path(output) / f"{filename}-dual.pdf"
        doc_mono = open(file_mono, "wb")
        doc_dual = open(file_dual, "wb")
        doc_mono.write(s_mono)
        doc_dual.write(s_dual)
        doc_mono.close()
        doc_dual.close()
        if s_debug:
            file_debug = Path(output) / f"{filename}-debug.pdf"
            doc_debug = open(file_debug, "wb")
            doc_debug.write(s_debug)
            doc_debug.close()
        result_files.append((str(file_mono), str(file_dual)))

    return result_files


def download_remote_fonts(lang: str):
    lang = lang.lower()
    LANG_NAME_MAP = {
        **{la: "GoNotoKurrent-Regular.ttf" for la in noto_list},
        **{
            la: f"SourceHanSerif{region}-Regular.ttf"
            for region, langs in {
                "CN": ["zh-cn", "zh-hans", "zh"],
                "TW": ["zh-tw", "zh-hant"],
                "JP": ["ja"],
                "KR": ["ko"],
            }.items()
            for la in langs
        },
    }
    font_name = LANG_NAME_MAP.get(lang, "GoNotoKurrent-Regular.ttf")

    # docker
    font_path = ConfigManager.get("NOTO_FONT_PATH", Path("/app", font_name).as_posix())
    if not Path(font_path).exists():
        font_path, _ = get_font_and_metadata(font_name)
        font_path = font_path.as_posix()

    logger.info(f"use font: {font_path}")

    return font_path
