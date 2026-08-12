"""winnow.py — WiNNoW의 몸통.

여기까지 구현됨:
  ①   폴더 안의 사진을 찾아 파일명과 촬영 시각(EXIF)을 읽는다.
  ①-b 사진 한 장을 그림으로 연다 — RAW든 JPEG든 load_image() 하나로.
  ②   촬영 시각으로 묶고, 닮은 묶음끼리 다시 합친다.

아직 없는 것: 흐림 점수(③) · keep/reject/애매 분류(④) · 폴더로 복사(⑤).
"""

import io
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

import exifread
import numpy as np
import rawpy
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

# ------------------------------------------------------------
# 묶기 설정값
# ------------------------------------------------------------
# 사진 사이 시간 간격이 이보다 크면 다른 묶음으로 나눈다. (초)
#   작게 하면 연사만 묶이고, 크게 하면 한 장면을 넓게 묶는다.
#   나뉜 뒤 닮은 묶음끼리는 어차피 다시 합쳐지므로 조금 작게 잡아도 된다.
TIME_GAP_SECONDS = 10

# 두 사진의 지문이 이만큼 이하로 다르면 "닮았다"고 보고 묶음을 합친다.
#   지문은 64칸이라 0이면 똑같고 64면 완전히 다르다.
#   0~5 = 거의 같은 컷, 10 = 비슷한 구도, 20 이상 = 남남에 가까움.
HASH_DISTANCE_MAX = 10

# ------------------------------------------------------------
# RAW를 그림으로 여는 방식 (주 대상이 RAW라 중요)
# ------------------------------------------------------------
# RAW 안에는 카메라가 만들어 넣은 JPEG 미리보기가 있다. 기본은 그걸 꺼내 쓴다.
# True로 바꾸면 미리보기를 무시하고 직접 현상한다.
#   느리지만(400장에 10분 vs 1분) 카메라의 노이즈 제거·샤프닝이 안 걸린
#   순수한 그림으로 판정한다. 흐림 판정이 이상하다 싶으면 켜서 비교해 볼 것.
RAW_ALWAYS_DEMOSAIC = False

# 미리보기 가로폭이 이보다 작으면 못 믿고 직접 현상한다.
#   캐논은 원본 크기(5184px)로 넣어주지만 소니·파나소닉은 작게 넣는 편이다.
RAW_PREVIEW_MIN_WIDTH = 1000

# 판정에 쓸 그림의 긴 변 길이. 원본 크기 그대로 다룰 이유가 없다.
#   유사도는 어차피 9x8로 뭉개고, 흐림도 이 정도면 충분하다.
WORK_MAX_SIDE = 1024

# ※ 분류용 설정값(keep 비율·흐림 경계)은 해당 로직을 만드는
#   다음 단계에서 이 자리에 추가한다.

# 지문(dHash) 격자 크기. 가로로 이웃과 비교하므로 폭이 1 더 크다 → 8x8 = 64칸.
_HASH_W, _HASH_H = 9, 8


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

    # 그림에서 뽑은 지문(dHash). 64칸짜리 True/False 배열.
    # compare=False: 넘파이 배열은 ==로 비교하면 배열이 나와서 dataclass가 헷갈린다.
    fingerprint: np.ndarray | None = field(default=None, compare=False, repr=False)


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


# ============================================================
# ①-b 사진을 그림으로 열기
#
# RAW는 그냥 열리지 않는다. JPEG가 "완성된 그림"이라면 RAW는 센서가 받은
# 날것의 숫자라, 사람이 볼 그림으로 바꾸는 '현상'을 거쳐야 한다.
# 그래서 여는 통로를 load_image() 하나로 통일한다.
# 이 아래 로직들은 파일이 RAW인지 아닌지 몰라도 된다.
# ============================================================

def _shrink(image: Image.Image) -> Image.Image:
    """긴 변을 WORK_MAX_SIDE에 맞춘다. 이미 작으면 그대로 둔다."""
    w, h = image.size
    if max(w, h) <= WORK_MAX_SIDE:
        return image
    scale = WORK_MAX_SIDE / max(w, h)
    return image.resize((max(1, round(w * scale)), max(1, round(h * scale))))


def _demosaic(path: Path) -> Image.Image:
    """RAW를 직접 현상한다. 느린 길 (장당 약 1.5초).

    half_size=True — 가로세로 절반으로 현상. 어차피 줄여 쓸 거라
    시간은 비슷해도 메모리를 훨씬 덜 먹는다.
    """
    with rawpy.imread(str(path)) as raw:
        return Image.fromarray(raw.postprocess(half_size=True))


def _load_raw(path: Path) -> Image.Image:
    """RAW를 그림으로 연다. 기본은 내장 미리보기, 안 되면 현상.

    미리보기가 빠른 이유:
      draft()로 "어차피 작게 쓸 거니 1/4 크기로 풀어달라"고 요청한다.
      JPEG는 원래 그런 식으로 풀 수 있어서, 푸는 계산 자체가 줄어든다.
      실측 결과 4배 빨라지는데 판정 결과는 사실상 같았다.
    """
    if RAW_ALWAYS_DEMOSAIC:
        return _demosaic(path)

    try:
        with rawpy.imread(str(path)) as raw:
            thumb = raw.extract_thumb()
    except Exception:
        # 미리보기가 없거나 못 읽는 RAW — 현상으로 넘어간다.
        return _demosaic(path)

    if thumb.format != rawpy.ThumbFormat.JPEG:
        # 드물게 JPEG가 아닌 날 픽셀로 들어있는 경우.
        return _shrink(Image.fromarray(thumb.data))

    image = Image.open(io.BytesIO(thumb.data))
    if image.size[0] < RAW_PREVIEW_MIN_WIDTH:
        # 너무 작은 미리보기는 못 믿는다.
        return _demosaic(path)

    image.draft("RGB", (WORK_MAX_SIDE, WORK_MAX_SIDE))
    return _shrink(image.convert("RGB"))


def load_image(path: str | Path) -> Image.Image | None:
    """사진 한 장을 판정용 그림으로 연다. RAW든 JPEG든 이 함수 하나로.

    - 긴 변이 WORK_MAX_SIDE 이하로 줄어든 상태로 돌려준다.
    - 열 수 없는(깨진) 파일이면 None.
    """
    path = Path(path)
    try:
        if path.suffix.lower() in RAW_EXTENSIONS:
            return _load_raw(path)
        image = Image.open(path)
        image.draft("RGB", (WORK_MAX_SIDE, WORK_MAX_SIDE))
        return _shrink(image.convert("RGB"))
    except Exception:
        return None


# ============================================================
# ② 묶기 — 촬영 시각으로 나눈 뒤, 닮은 묶음끼리 다시 합치기
# ============================================================

def _dhash(image: Image.Image) -> np.ndarray:
    """그림의 지문을 뽑는다 (dHash).

    사진을 9x8 픽셀로 확 줄이고 흑백으로 만든 뒤,
    픽셀마다 "오른쪽 이웃보다 밝은가?"를 True/False로 적는다 → 64칸.
    색·밝기가 조금 달라도 구도가 같으면 지문이 비슷하게 나온다.
    """
    small = image.convert("L").resize((_HASH_W, _HASH_H))
    pixels = np.asarray(small, dtype=np.int16)
    return (pixels[:, 1:] > pixels[:, :-1]).flatten()


def _distance(a: np.ndarray, b: np.ndarray) -> int:
    """두 지문에서 다른 칸이 몇 개인지. 0이면 똑같고 64면 완전히 다르다."""
    return int(np.count_nonzero(a != b))


def compute_fingerprints(photos: list[Photo], on_progress=None) -> None:
    """사진마다 지문을 뽑아 Photo에 채워 넣는다. 여기가 제일 오래 걸린다.

    이미 지문이 있는 사진은 건너뛴다 (같은 목록으로 두 번 불러도 낭비가 없게).
    on_progress(끝난 장수, 전체 장수) — 진행 표시가 필요하면 넘긴다 (나중에 UI가 씀).
    """
    todo = [p for p in photos if p.fingerprint is None]
    for i, photo in enumerate(todo, start=1):
        image = load_image(photo.path)
        photo.fingerprint = _dhash(image) if image is not None else None
        if on_progress:
            on_progress(i, len(todo))


def _split_by_time(photos: list[Photo]) -> list[list[Photo]]:
    """촬영 시각으로 크게 나눈다. 간격이 TIME_GAP_SECONDS를 넘으면 끊는다.

    촬영 시각이 없는 사진은 시간으로는 판단할 수 없으니 각자 혼자 둔다.
    (뒤의 유사도 합치기에서 닮은 묶음과 붙을 기회가 있다.)
    """
    timed = sorted(
        (p for p in photos if p.taken_at is not None), key=lambda p: p.taken_at
    )
    untimed = [p for p in photos if p.taken_at is None]

    groups: list[list[Photo]] = []
    for photo in timed:
        if groups:
            gap = (photo.taken_at - groups[-1][-1].taken_at).total_seconds()
            if gap <= TIME_GAP_SECONDS:
                groups[-1].append(photo)
                continue
        groups.append([photo])

    groups.extend([p] for p in untimed)
    return groups


def _merge_by_similarity(groups: list[list[Photo]]) -> list[list[Photo]]:
    """닮은 사진이 든 묶음끼리 합친다.

    묶음 A의 사진 하나와 묶음 B의 사진 하나가 닮았으면 A와 B를 한 묶음으로 본다.
    시간이 떨어져 있어도 합쳐지므로, 같은 장면을 나중에 다시 찍은 것도 잡힌다.

    합치는 방법은 '이어달리기'다. A-B가 닮고 B-C가 닮으면 A-B-C가 한 묶음이 된다.
    """
    # 지문이 있는 사진만 비교할 수 있다. (사진, 그 사진이 속한 묶음 번호)
    items = [
        (photo, gi)
        for gi, group in enumerate(groups)
        for photo in group
        if photo.fingerprint is not None
    ]
    if len(items) < 2:
        return groups

    # 어느 묶음과 어느 묶음이 한 덩어리인지 추적한다 (합집합 찾기).
    parent = list(range(len(groups)))

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]   # 지나가는 길에 지름길을 만들어 둔다
            x = parent[x]
        return x

    def union(a: int, b: int) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[max(ra, rb)] = min(ra, rb)

    # 모든 사진 쌍을 견준다. 넘파이로 한 줄씩 한꺼번에 계산한다.
    matrix = np.array([photo.fingerprint for photo, _ in items])
    owners = np.array([gi for _, gi in items])
    for i in range(len(items) - 1):
        distances = np.count_nonzero(matrix[i + 1:] != matrix[i], axis=1)
        for offset in np.flatnonzero(distances <= HASH_DISTANCE_MAX):
            union(owners[i], owners[i + 1 + offset])

    # 한 덩어리가 된 묶음들을 실제로 합친다. 순서는 원래 순서를 따른다.
    merged: dict[int, list[Photo]] = {}
    for gi, group in enumerate(groups):
        merged.setdefault(find(gi), []).extend(group)

    result = list(merged.values())
    for group in result:
        # 묶음 안은 촬영 시각 순, 시각이 없는 건 뒤로.
        group.sort(key=lambda p: (p.taken_at is None, p.taken_at or datetime.min, p.name))
    return result


def group_photos(photos: list[Photo], on_progress=None) -> list[list[Photo]]:
    """사진 목록을 묶음들로 나눈다. ②단계의 최종 결과.

    1) 촬영 시각으로 크게 나눈다
    2) 사진마다 지문을 뽑는다  ← 오래 걸리는 곳
    3) 닮은 묶음끼리 다시 합친다
    """
    if not photos:
        return []

    groups = _split_by_time(photos)
    compute_fingerprints(photos, on_progress)
    return _merge_by_similarity(groups)


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
    import time

    if len(sys.argv) < 2:
        print('사용법: python winnow.py "폴더경로"')
        sys.exit(1)

    target = sys.argv[1]
    found = read_photos(target)
    print(f"'{target}' 에서 사진 {len(found)}장을 찾았습니다.")

    if not found:
        sys.exit(0)

    with_time = sum(1 for p in found if p.taken_at)
    print(f"촬영 시각이 있는 사진: {with_time} / {len(found)}장\n")

    def show_progress(done: int, total: int) -> None:
        print(f"\r  지문 뽑는 중... {done}/{total}", end="", flush=True)

    started = time.perf_counter()
    groups = group_photos(found, on_progress=show_progress)
    elapsed = time.perf_counter() - started
    print(f"\r  지문 뽑기 완료 ({elapsed:.1f}초, 장당 {elapsed/len(found):.3f}초)\n")

    print(f"=== 묶음 {len(groups)}개 ===\n")
    for i, group in enumerate(groups, start=1):
        mark = "  ← 여러 장" if len(group) > 1 else ""
        print(f"[묶음 {i}] {len(group)}장{mark}")
        for photo in group:
            when = photo.taken_at.strftime("%H:%M:%S") if photo.taken_at else "시각 없음"
            print(f"     {photo.name:<20} {when}")
        print()

    alone = sum(1 for g in groups if len(g) == 1)
    print(f"요약: 사진 {len(found)}장 → 묶음 {len(groups)}개 "
          f"(혼자인 묶음 {alone}개, 여러 장인 묶음 {len(groups) - alone}개)")
    print(f"설정: 시간 간격 {TIME_GAP_SECONDS}초, 유사도 경계 {HASH_DISTANCE_MAX}")
