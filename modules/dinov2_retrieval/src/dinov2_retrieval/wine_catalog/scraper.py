"""Polite HTTP scraper for vino-svoe.ru wine cards and reference photos."""

from __future__ import annotations

import hashlib
import html
import json
import os
import re
import sys
import time
from dataclasses import dataclass
from html.parser import HTMLParser
from io import BytesIO
from pathlib import Path
from typing import Any
from urllib.parse import quote, urljoin, urlparse
from urllib.robotparser import RobotFileParser

from .migrations import apply_migrations

SITE_ORIGIN = "https://vino-svoe.ru"
DEFAULT_TIMEOUT_SECONDS = 20.0
DEFAULT_MIN_DELAY_SECONDS = 1.0
DEFAULT_RETRIES = 3
MAX_IMAGE_BYTES = 30 * 1024 * 1024
IMAGE_EXTENSIONS = {
    "JPEG": ".jpg",
    "PNG": ".png",
    "WEBP": ".webp",
    "GIF": ".gif",
    "TIFF": ".tiff",
    "BMP": ".bmp",
}
CHALLENGE_MARKERS = (
    "captcha",
    "капча",
    "проверка что вы не робот",
    "подтвердите, что вы не робот",
    "доступ ограничен",
)


class ScrapeError(RuntimeError):
    """A card or its reference image could not be safely extracted."""


class ChallengeDetected(ScrapeError):
    """The site requires a human check before returning the card."""


class PageNotRendered(ScrapeError):
    """The response has no parsed product data and may need a browser."""


@dataclass(frozen=True)
class WinePage:
    image_url: str


class _WineHTMLParser(HTMLParser):
    """Collect semantic page data without relying on guessed CSS selectors."""

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.meta: dict[str, str] = {}
        self.images: list[dict[str, str]] = []
        self.json_ld: list[str] = []
        self.text_parts: list[str] = []
        self.headings: list[tuple[str, str]] = []
        self._heading_tag: str | None = None
        self._heading_parts: list[str] = []
        self._script_type: str | None = None
        self._script_parts: list[str] = []
        self._ignored_depth = 0

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        attrs_dict = {key.lower(): value or "" for key, value in attrs}
        if tag.lower() == "meta":
            key = attrs_dict.get("property") or attrs_dict.get("name")
            if key and attrs_dict.get("content"):
                self.meta[key.casefold()] = attrs_dict["content"].strip()
        elif tag.lower() in ("img", "source"):
            self.images.append(attrs_dict)

        if tag.lower() == "script":
            self._script_type = attrs_dict.get("type", "").casefold()
            self._script_parts = []
        elif tag.lower() in ("style", "noscript"):
            self._ignored_depth += 1

        if tag.lower() in ("h1", "h2", "h3"):
            self._heading_tag = tag.lower()
            self._heading_parts = []

    def handle_endtag(self, tag: str) -> None:
        tag = tag.lower()
        if tag == "script":
            if "ld+json" in (self._script_type or ""):
                self.json_ld.append("".join(self._script_parts))
            self._script_type = None
            self._script_parts = []
        elif tag in ("style", "noscript") and self._ignored_depth:
            self._ignored_depth -= 1

        if tag == self._heading_tag and self._heading_tag:
            heading = _collapse(" ".join(self._heading_parts))
            if heading:
                self.headings.append((tag, heading))
            self._heading_tag = None
            self._heading_parts = []

    def handle_data(self, data: str) -> None:
        if self._script_type:
            self._script_parts.append(data)
            return
        if self._ignored_depth:
            return
        value = data.strip()
        if value:
            self.text_parts.append(value)
            if self._heading_tag:
                self._heading_parts.append(value)


def parse_wine_page(page_html: str, page_url: str) -> WinePage:
    """Extract a product image; wine characteristics are imported from CSV."""

    parser = _WineHTMLParser()
    parser.feed(page_html)
    body_text = _collapse(" ".join(parser.text_parts))
    lowered = body_text.casefold()
    if any(marker in lowered for marker in CHALLENGE_MARKERS):
        raise ChallengeDetected("Site returned a CAPTCHA or human verification page")

    structured = _structured_product(parser.json_ld)
    h1 = next((text for tag, text in parser.headings if tag == "h1"), None)
    name = _first_text(
        structured.get("name"), h1, parser.meta.get("og:title"), parser.meta.get("twitter:title")
    )

    bottle_images: list[str] = []
    exact_images: list[str] = []
    generic_images: list[str] = []
    for image_attrs in parser.images:
        candidate = (
            image_attrs.get("data-src")
            or image_attrs.get("data-original")
            or image_attrs.get("src")
            or image_attrs.get("srcset", "").split(",", 1)[0].split(" ", 1)[0]
        )
        alt = image_attrs.get("alt", "").casefold()
        classes = image_attrs.get("class", "").split()
        if candidate and any(item.endswith("__bottle") for item in classes):
            # Observed on actual vino-svoe.ru cards: the full bottle photo.
            bottle_images.append(candidate)
        elif candidate and name and name.casefold() in alt:
            exact_images.append(candidate)
        elif candidate and ("wine" in alt or "вино" in alt):
            generic_images.append(candidate)

    image_candidates = [
        *bottle_images,
        structured.get("image"),
        *exact_images,
        parser.meta.get("og:image"),
        parser.meta.get("twitter:image"),
        *generic_images,
    ]

    image_url = next(
        (
            urljoin(page_url, candidate.strip())
            for candidate in image_candidates
            if candidate and _is_http_url(urljoin(page_url, candidate.strip()))
        ),
        None,
    )
    if not image_url:
        raise PageNotRendered(
            "Wine card did not expose a product image in HTML/metadata; "
            "try --interactive if the card requires browser rendering"
        )
    return WinePage(image_url=image_url)


def scrape_wines(
    connection,
    *,
    slugs: list[str] | None = None,
    retry_failed: bool = False,
    limit: int | None = None,
    photos_dir: Path = Path("data/web_photos"),
    timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS,
    min_delay_seconds: float = DEFAULT_MIN_DELAY_SECONDS,
    retries: int = DEFAULT_RETRIES,
    interactive: bool = False,
    cookie_jar_path: Path | None = None,
    browser_profile: Path | None = None,
) -> dict[str, int]:
    """Process pending slugs, optionally limiting a run to a small sample."""

    if timeout_seconds <= 0 or min_delay_seconds < 0 or retries < 0:
        raise ValueError("timeout must be positive; delay and retries cannot be negative")
    photo_root = photos_dir.expanduser().resolve()
    photo_root.mkdir(parents=True, exist_ok=True)
    state_root = photo_root.parent
    cookie_path = (cookie_jar_path or state_root / ".wine-scraper-cookies.json").expanduser().resolve()
    cookie_path.parent.mkdir(parents=True, exist_ok=True)
    browser_profile = (browser_profile or state_root / ".wine-scraper-browser-profile").expanduser().resolve()
    browser_profile.mkdir(parents=True, exist_ok=True)

    import httpx

    apply_migrations(connection)
    connection.execute(
        "INSERT INTO wine_scrape_jobs (slug) "
        "SELECT DISTINCT slug FROM wines WHERE slug IS NOT NULL AND btrim(slug) <> '' "
        "ON CONFLICT (slug) DO NOTHING"
    )
    statuses = "status = 'failed'" if retry_failed else "status <> 'succeeded'"
    parameters: list[Any] = []
    sql = (
        "SELECT j.slug FROM wine_scrape_jobs j WHERE " + statuses
        + " AND EXISTS (SELECT 1 FROM wines w WHERE w.slug = j.slug)"
    )
    if slugs:
        sql += " AND j.slug = ANY(%s)"
        parameters.append(slugs)
    sql += " ORDER BY j.slug"
    if limit is not None:
        sql += " LIMIT %s"
        parameters.append(limit)
    candidates = [row[0] for row in connection.execute(sql, parameters).fetchall()]
    if slugs:
        known = set(
            row[0]
            for row in connection.execute(
                "SELECT DISTINCT slug FROM wines WHERE slug = ANY(%s)", (slugs,)
            ).fetchall()
        )
        missing = sorted(set(slugs) - known)
        if missing:
            raise ValueError("Slugs not found in imported wines table: " + ", ".join(missing))

    summary = {"processed_cards": 0, "downloaded_images": 0, "errors": 0}
    if not candidates:
        return summary
    user_agent = os.getenv(
        "WINE_SCRAPER_USER_AGENT", "SvoeVinO-wine-catalog-pipeline/1.0"
    )
    with httpx.Client(
        timeout=httpx.Timeout(timeout_seconds),
        follow_redirects=True,
        headers={"User-Agent": user_agent, "Accept": "text/html,application/xhtml+xml,image/*"},
    ) as client:
        _load_cookies(client, cookie_path)
        last_request_at = [0.0]
        try:
            robots = _read_robots(client, user_agent, retries, min_delay_seconds, last_request_at)
        except Exception as exc:
            for slug in candidates:
                connection.execute(
                    "UPDATE wine_scrape_jobs SET status = 'failed', attempts = attempts + 1, "
                    "last_error = %s, processed_at = NOW(), updated_at = NOW() WHERE slug = %s",
                    (f"Could not check robots.txt: {exc}"[:4000], slug),
                )
            summary["errors"] = len(candidates)
            return summary
        for slug in candidates:
            connection.execute(
                "UPDATE wine_scrape_jobs SET status = 'in_progress', attempts = attempts + 1, "
                "last_error = NULL, updated_at = NOW() WHERE slug = %s",
                (slug,),
            )
            try:
                card_url = f"{SITE_ORIGIN}/wines/{quote(slug, safe='-_.~')}"
                try:
                    if robots is not None and not robots.can_fetch(user_agent, card_url):
                        raise ScrapeError(f"robots.txt disallows {card_url}")
                    response = _request(client, card_url, retries, min_delay_seconds, last_request_at)
                    _check_response(response)
                    wine = parse_wine_page(response.text, str(response.url))
                except (ChallengeDetected, PageNotRendered):
                    if not interactive:
                        raise
                    page_html = _read_with_playwright(
                        card_url, browser_profile, client, cookie_path
                    )
                    wine = parse_wine_page(page_html, card_url)

                if (
                    robots is not None
                    and urlparse(wine.image_url).netloc == urlparse(SITE_ORIGIN).netloc
                    and not robots.can_fetch(user_agent, wine.image_url)
                ):
                    raise ScrapeError(f"robots.txt disallows {wine.image_url}")

                existing = connection.execute(
                    "SELECT web_photo FROM wines WHERE slug = %s AND web_photo IS NOT NULL "
                    "ORDER BY id LIMIT 1",
                    (slug,),
                ).fetchone()
                image_path = _find_existing_image(slug, existing[0] if existing else None, photo_root)
                image_downloaded = image_path is None
                if image_downloaded:
                    image_response = _request(
                        client, wine.image_url, retries, min_delay_seconds, last_request_at
                    )
                    _check_response(image_response)
                    image_path = _save_verified_image(slug, image_response.content, photo_root)

                relative_photo = _relative_photo_path(image_path, photos_dir)
                _save_success(connection, slug, wine, relative_photo)
                summary["processed_cards"] += 1
                if image_downloaded:
                    summary["downloaded_images"] += 1
            except Exception as exc:
                summary["errors"] += 1
                connection.execute(
                    "UPDATE wine_scrape_jobs SET status = 'failed', last_error = %s, "
                    "processed_at = NOW(), updated_at = NOW() WHERE slug = %s",
                    (str(exc)[:4000], slug),
                )
    return summary


def _request(client, url: str, retries: int, delay: float, last_request_at: list[float]):
    import httpx

    for attempt in range(retries + 1):
        remaining = delay - (time.monotonic() - last_request_at[0])
        if remaining > 0:
            time.sleep(remaining)
        last_request_at[0] = time.monotonic()
        try:
            response = client.get(url)
        except httpx.HTTPError:
            if attempt >= retries:
                raise
            time.sleep(min(30.0, 2**attempt))
            continue
        if response.status_code == 429 or 500 <= response.status_code < 600:
            if attempt >= retries:
                return response
            retry_after = response.headers.get("Retry-After")
            try:
                backoff = min(60.0, float(retry_after)) if retry_after else 2**attempt
            except ValueError:
                backoff = 2**attempt
            time.sleep(backoff)
            continue
        return response
    raise AssertionError("unreachable")


def _check_response(response) -> None:
    text = response.text[:100_000].casefold() if "text" in response.headers.get("content-type", "") else ""
    if response.status_code in (401, 403) or any(marker in text for marker in CHALLENGE_MARKERS):
        raise ChallengeDetected(
            f"Human verification or access restriction returned HTTP {response.status_code}; "
            "rerun with --interactive to complete it in a persistent browser profile"
        )
    response.raise_for_status()


def _read_robots(client, user_agent: str, retries: int, delay: float, last_request_at: list[float]):
    robots_url = f"{SITE_ORIGIN}/robots.txt"
    response = _request(client, robots_url, retries, delay, last_request_at)
    if response.status_code == 404:
        return None
    _check_response(response)
    parser = RobotFileParser(robots_url)
    parser.parse(response.text.splitlines())
    return parser


def _save_success(connection, slug: str, wine: WinePage, relative_photo: str) -> None:
    with connection.transaction():
        connection.execute(
            """
            UPDATE wines SET
                web_photo = %(web_photo)s,
                web_photo_url = %(web_photo_url)s
            WHERE slug = %(slug)s
            """,
            {
                "web_photo": relative_photo,
                "web_photo_url": wine.image_url,
                "slug": slug,
            },
        )
        connection.execute(
            "UPDATE wine_scrape_jobs SET status = 'succeeded', last_error = NULL, "
            "processed_at = NOW(), updated_at = NOW() WHERE slug = %s",
            (slug,),
        )


def _save_verified_image(slug: str, data: bytes, photo_root: Path) -> Path:
    try:
        from PIL import Image, UnidentifiedImageError
    except ImportError as exc:
        raise ScrapeError("Image validation requires Pillow; install the project dependencies") from exc

    if not data or len(data) > MAX_IMAGE_BYTES:
        raise ScrapeError(f"Downloaded image has invalid size: {len(data)} bytes")
    try:
        with Image.open(BytesIO(data)) as image:
            image_format = (image.format or "").upper()
            if image.width < 100 or image.height < 100:
                raise ScrapeError("Downloaded image is too small to be a wine reference photo")
            image.verify()
    except (UnidentifiedImageError, OSError, ValueError) as exc:
        raise ScrapeError("Downloaded response is not a valid image file") from exc
    extension = IMAGE_EXTENSIONS.get(image_format)
    if not extension:
        raise ScrapeError(f"Unsupported image format: {image_format or 'unknown'}")
    stem = _photo_stem(slug)
    target = photo_root / f"{stem}{extension}"
    temporary = photo_root / f".{stem}.part"
    temporary.write_bytes(data)
    temporary.replace(target)
    return target


def _find_existing_image(slug: str, stored_path: str | None, photo_root: Path) -> Path | None:
    from PIL import Image, UnidentifiedImageError

    stem = _photo_stem(slug)
    candidates = [photo_root / f"{stem}{extension}" for extension in IMAGE_EXTENSIONS.values()]
    if stored_path:
        candidates.insert(0, photo_root / Path(stored_path).name)
    for candidate in candidates:
        if candidate.is_file():
            try:
                with Image.open(candidate) as image:
                    image.verify()
                return candidate
            except (UnidentifiedImageError, OSError, ValueError):
                continue
    return None


def _relative_photo_path(photo_path: Path, photos_dir: Path) -> str:
    configured = photos_dir.expanduser()
    if not configured.is_absolute():
        return (configured / photo_path.name).as_posix()
    return photo_path.name


def _photo_stem(slug: str) -> str:
    safe = re.sub(r"[^A-Za-z0-9._-]+", "_", slug).strip("._-")[:80] or "wine"
    digest = hashlib.sha256(slug.encode("utf-8")).hexdigest()[:10]
    return f"{safe}-{digest}"


def _structured_product(scripts: list[str]) -> dict[str, str | None]:
    products: list[dict[str, Any]] = []
    for script in scripts:
        try:
            value = json.loads(script)
        except (json.JSONDecodeError, TypeError):
            continue
        products.extend(_walk_product_objects(value))
    if not products:
        return {}
    product = products[0]
    image = product.get("image")
    if isinstance(image, list):
        image = next((item for item in image if isinstance(item, str)), None)
    elif isinstance(image, dict):
        image = image.get("url") or image.get("contentUrl")
    return {
        "name": _as_text(product.get("name")),
        "image": _as_text(image),
    }


def _walk_product_objects(value: Any):
    if isinstance(value, dict):
        kind = value.get("@type", "")
        types = kind if isinstance(kind, list) else [kind]
        if any(str(item).casefold() in {"product", "wine"} for item in types):
            yield value
        for child in value.values():
            yield from _walk_product_objects(child)
    elif isinstance(value, list):
        for child in value:
            yield from _walk_product_objects(child)


def _read_with_playwright(card_url: str, profile: Path, client, cookie_path: Path) -> str:
    if not sys.stdin.isatty():
        raise ScrapeError("Interactive challenge recovery requires a terminal")
    try:
        from playwright.sync_api import sync_playwright
    except ImportError as exc:
        raise ScrapeError(
            "Install the optional browser extra with `pip install -e '.[catalog-browser]'` "
            "and `playwright install chromium`"
        ) from exc
    with sync_playwright() as playwright:
        context = playwright.chromium.launch_persistent_context(
            user_data_dir=str(profile), headless=False
        )
        try:
            page = context.pages[0] if context.pages else context.new_page()
            page.goto(card_url, wait_until="domcontentloaded", timeout=30_000)
            input(
                "Проверьте страницу в браузере, вручную пройдите CAPTCHA/подтвердите возраст "
                "при необходимости, затем нажмите Enter для продолжения... "
            )
            page_html = page.content()
            for cookie in context.cookies():
                domain = cookie.get("domain") or ""
                if not domain.lstrip(".").endswith("vino-svoe.ru"):
                    continue
                client.cookies.set(
                    cookie["name"], cookie["value"],
                    domain=domain, path=cookie.get("path", "/"),
                )
            _save_cookies(client, cookie_path)
            return page_html
        finally:
            context.close()


def _load_cookies(client, path: Path) -> None:
    if not path.is_file():
        return
    try:
        cookies = json.loads(path.read_text(encoding="utf-8"))
        for cookie in cookies:
            client.cookies.set(
                cookie["name"], cookie["value"],
                domain=cookie.get("domain"), path=cookie.get("path", "/"),
            )
    except (OSError, json.JSONDecodeError, KeyError, TypeError):
        return


def _save_cookies(client, path: Path) -> None:
    cookies = [
        {
            "name": cookie.name,
            "value": cookie.value,
            "domain": cookie.domain,
            "path": cookie.path,
        }
        for cookie in client.cookies.jar
    ]
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(descriptor, "w", encoding="utf-8") as output:
        json.dump(cookies, output, ensure_ascii=False)
    try:
        path.chmod(0o600)
    except OSError:
        pass


def _is_http_url(value: str) -> bool:
    return urlparse(value).scheme in ("http", "https")


def _as_text(value: Any) -> str | None:
    return _clean_capture(str(value)) if value is not None and not isinstance(value, (dict, list)) else None


def _first_text(*values: str | None) -> str | None:
    return next((cleaned for value in values if (cleaned := _clean_capture(value))), None)


def _clean_capture(value: str | None) -> str | None:
    if value is None:
        return None
    cleaned = _collapse(html.unescape(value)).strip(" :\u00a0\t\r\n")
    return cleaned or None


def _collapse(value: str) -> str:
    return re.sub(r"\s+", " ", value).strip()
