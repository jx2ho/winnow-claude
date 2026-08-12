"""winnow.py — WiNNoW의 몸통.

지금은 v1 1단계만 구현: 폴더 안의 사진을 찾아 파일명과 촬영 시각(EXIF)을 읽는다.
묶기·흐림 판정·분류·복사는 아직 없다 (다음 단계).
"""

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import exifread
from PIL import Image

# HEIC(아이폰 사진)도 열 수 있게 등록. 안 깔려 있어도 나머지는 돌아간다.
try:
    from pillow_heif import register_heif_opener

    register_heif_opener()
except ImportError:
    pass


# ============================================================
# 조절 설정값 (여기만 바꾸면 됨)
# ============================================================
# Pillow로 여는 일반 사진 확장자. 소문자로 적는다 (비교할 때 소문자로 맞춤).
STANDARD_EXTENSIONS = {
    ".jpg", ".jpeg", ".png", ".webp",
    ".bmp", ".tiff", ".tif", ".gif", ".heic",
}

# RAW(카메라 날것) 확장자. Pillow로는 못 열어서 exifread로 촬영 시각만 읽는다.
#   cr2/cr3=캐논, nef=니콘, arw=소니, dng=어도비, orf=올림푸스,
#   rw2=파나소닉, raf=후지, pef=펜탁스, srw=삼성
# ※ 지금은 '촬영 시각'만 읽는다. RAW의 실제 픽셀(흐림·유사도용)은 나중 단계에서.
RAW_EXTENSIONS = {
    ".cr2", ".cr3", ".nef", ".arw", ".dng",
    ".orf", ".rw2", ".raf", ".pef", ".srw",
}

# '사진'으로 볼 확장자 전체 (일반 + RAW).
PHOTO_EXTENSIONS = STANDARD_EXTENSIONS | RAW_EXTENSIONS

# ※ 분류용 설정값 세 개(keep 비율·유사도 경계·흐림 경계)는
#   해당 로직을 만드는 다음 단계에서 이 자리에 추가한다.


# EXIF 태그 번호 (Pillow가 숫자로 준다)
_EXIF_DATETIME_ORIGINAL = 36867   # 촬영한 순간 (제일 정확)
_EXIF_DATETIME_DIGITIZED = 36868  # 디지털로 저장된 순간 (차선)
_EXIF_DATETIME = 306              # 파일이 마지막으로 수정된 순간 (최후)
_EXIF_SUB_IFD = 0x8769            # DateTimeOriginal 등이 들어있는 하위 묶음


@dataclass
class Photo:
    """사진 한 장의 정보."""
    path: Path              # 파일의 전체 경로
    name: str               # 파일명 (예: IMG_1234.jpg)
    taken_at: datetime | None  # 촬영 시각. 없으면 None


def _parse_exif_datetime(value) -> datetime | None:
    """EXIF의 시각 문자열('2026:08:13 14:30:00')을 datetime으로 바꾼다.

    형식이 이상하거나 값이 없으면 None을 돌려준다.
    """
    if not value:
        return None
    try:
        return datetime.strptime(str(value).strip(), "%Y:%m:%d %H:%M:%S")
    except ValueError:
        return None


def _read_taken_at(image: Image.Image) -> datetime | None:
    """열린 이미지에서 촬영 시각을 최선을 다해 찾는다.

    DateTimeOriginal(촬영) → DateTimeDigitized(저장) → DateTime(수정) 순으로 시도.
    셋 다 없으면 None.
    """
    exif = image.getexif()
    if not exif:
        return None

    # DateTimeOriginal / DateTimeDigitized 는 하위 묶음(Exif IFD) 안에 있다.
    sub = exif.get_ifd(_EXIF_SUB_IFD)
    for tag in (_EXIF_DATETIME_ORIGINAL, _EXIF_DATETIME_DIGITIZED):
        taken = _parse_exif_datetime(sub.get(tag))
        if taken:
            return taken

    # 마지막으로 기본 IFD의 DateTime.
    return _parse_exif_datetime(exif.get(_EXIF_DATETIME))


def _read_taken_at_raw(path: Path) -> datetime | None:
    """RAW 파일(CR2 등)에서 exifread로 촬영 시각을 찾는다.

    DateTimeOriginal(촬영) → DateTimeDigitized(저장) → DateTime(수정) 순으로 시도.
    못 읽으면 None.
    """
    try:
        with open(path, "rb") as f:
            tags = exifread.process_file(f, details=False)
    except (OSError, ValueError):
        return None

    for key in ("EXIF DateTimeOriginal", "EXIF DateTimeDigitized", "Image DateTime"):
        taken = _parse_exif_datetime(tags.get(key))
        if taken:
            return taken
    return None


def read_photo(path: Path) -> Photo:
    """사진 파일 하나를 읽어 Photo로 만든다.

    - RAW 파일은 exifread로, 나머지는 Pillow로 촬영 시각을 읽는다.
    - 파일이 깨졌거나 시각을 못 읽으면 taken_at을 None으로 두고 넘어간다.
    """
    taken_at = None
    if path.suffix.lower() in RAW_EXTENSIONS:
        taken_at = _read_taken_at_raw(path)
    else:
        try:
            with Image.open(path) as image:
                taken_at = _read_taken_at(image)
        except (OSError, ValueError):
            # 열 수 없는(깨진) 파일 — 시각 없이 목록에는 남긴다.
            taken_at = None
    return Photo(path=path, name=path.name, taken_at=taken_at)


def read_photos(folder: str | Path) -> list[Photo]:
    """폴더 안의 모든 사진을 찾아 Photo 목록으로 돌려준다.

    - 하위 폴더까지 훑는다.
    - 사진이 아닌 파일은 건너뛴다.
    - 파일명 순으로 정렬해서 돌려준다.
    """
    folder = Path(folder)
    if not folder.is_dir():
        raise NotADirectoryError(f"폴더가 아니거나 없는 경로입니다: {folder}")

    photos: list[Photo] = []
    for path in folder.rglob("*"):
        if path.is_file() and path.suffix.lower() in PHOTO_EXTENSIONS:
            photos.append(read_photo(path))

    photos.sort(key=lambda p: p.name.lower())
    return photos


if __name__ == "__main__":
    # 터미널에서 테스트하는 실행부.
    #   python winnow.py "폴더경로"
    import sys

    if len(sys.argv) < 2:
        print('사용법: python winnow.py "폴더경로"')
        sys.exit(1)

    target = sys.argv[1]
    found = read_photos(target)

    print(f"'{target}' 에서 사진 {len(found)}장을 찾았습니다.\n")

    with_time = 0
    for i, photo in enumerate(found, start=1):
        if photo.taken_at:
            when = photo.taken_at.strftime("%Y-%m-%d %H:%M:%S")
            with_time += 1
        else:
            when = "촬영 시각 없음"
        print(f"{i:>3}. {photo.name:<40} {when}")

    print(f"\n촬영 시각이 있는 사진: {with_time} / {len(found)}장")
