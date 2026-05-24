from __future__ import annotations

import re
from concurrent.futures import ThreadPoolExecutor
from urllib.parse import quote_plus

import requests

from src.models import PriceOffer


SCRAPINGDOG_API_KEY = "6a13405f25544cb4f6333e26"
SCRAPINGDOG_SEARCH_URL = "https://api.scrapingdog.com/amazon/search"
SCRAPINGDOG_OFFERS_URL = "https://api.scrapingdog.com/amazon/offers"
SCRAPINGDOG_GOOGLE_SHOPPING_URL = "https://api.scrapingdog.com/google_shopping"
AMAZON_DOMAIN = "in"
AMAZON_COUNTRY = "in"
AMAZON_POSTAL_CODE = "411003"
AMAZON_LOCATION_LABEL = "Delivering to Pune 411003"
BOX_SET_HINTS = ("boxed set", "box set", "complete", "collection", "series")
GOOGLE_LANGUAGE = "en"


class PriceService:
    def search_book_prices(self, query: str, limit: int = 3) -> list[PriceOffer]:
        if not query.strip():
            return []

        try:
            with ThreadPoolExecutor(max_workers=2) as executor:
                amazon_future = executor.submit(self._search_amazon_products, query)
                google_future = executor.submit(self._search_google_shopping, query)
                amazon_payload = amazon_future.result()
                google_payload = google_future.result()
        except Exception:
            return self._fallback_search_offers(
                query, "Price lookup service is temporarily unavailable."
            )

        amazon_products = self._rank_products(
            query, self._extract_products(amazon_payload)
        )
        google_offers = self._extract_google_shopping_offers(google_payload, limit)

        if not amazon_products and not google_offers:
            return self._fallback_search_offers(query, "No direct price results found.")

        offers = self._build_amazon_offers(query, amazon_products[:limit])
        combined = self._merge_offers(offers, google_offers, limit)
        if not combined:
            return self._fallback_search_offers(query, "No direct price results found.")
        return combined

    def _build_amazon_offers(self, query: str, products: list[dict]) -> list[PriceOffer]:
        if not products:
            return []

        with ThreadPoolExecutor(max_workers=min(4, len(products))) as executor:
            product_details = list(
                executor.map(
                    lambda product: self._fetch_product_offers(
                        self._optional_text(product.get("asin")) or ""
                    ),
                    products,
                )
            )

        offers: list[PriceOffer] = []
        for product, product_details in zip(products, product_details, strict=False):
            asin = self._optional_text(product.get("asin"))
            product_url = self._optional_text(
                product_details.get("link") if isinstance(product_details, dict) else None
            ) or self._optional_text(
                product.get("optimized_url")
                or product.get("url")
                or product.get("link")
                or product.get("product_url")
            ) or self._build_product_url(asin)
            image_url = self._optional_text(
                product_details.get("image") if isinstance(product_details, dict) else None
            ) or self._optional_text(
                product.get("img_url")
                or product.get("image")
                or product.get("image_url")
                or product.get("thumbnail")
            )
            best_offer = self._extract_primary_offer(product_details)
            price = self._extract_price(best_offer) or self._optional_text(
                product.get("price_string")
                or product.get("price")
                or product.get("final_price")
                or product.get("display_price")
            )
            offers.append(
                PriceOffer(
                    source="amazon",
                    provider="Amazon",
                    title=self._optional_text(product.get("title")) or query,
                    price=price,
                    currency=self._extract_currency(best_offer)
                    or self._optional_text(product.get("currency"))
                    or self._optional_text(product.get("price_symbol")),
                    image_url=self._normalize_image_url(image_url),
                    asin=asin,
                    product_url=product_url,
                    note=self._build_offer_note(product, best_offer),
                )
            )
        return offers

    def _search_amazon_products(self, query: str) -> dict | None:
        response = requests.get(
            SCRAPINGDOG_SEARCH_URL,
            params={
                "api_key": SCRAPINGDOG_API_KEY,
                "query": query,
                "domain": AMAZON_DOMAIN,
                "page": 1,
                "country": AMAZON_COUNTRY,
                "postal_code": AMAZON_POSTAL_CODE,
                "premium": "false",
                "combined_output": "true",
                "type": "books",
            },
            timeout=30,
        )
        if response.status_code != 200:
            return None
        payload = response.json()
        return payload if isinstance(payload, dict) else None

    def _fetch_product_offers(self, asin: str) -> dict | None:
        if not asin:
            return None
        try:
            response = requests.get(
                SCRAPINGDOG_OFFERS_URL,
                params={
                    "api_key": SCRAPINGDOG_API_KEY,
                    "asin": asin,
                    "domain": AMAZON_DOMAIN,
                    "country": AMAZON_COUNTRY,
                    "postal_code": AMAZON_POSTAL_CODE,
                },
                timeout=30,
            )
            if response.status_code != 200:
                return None
            payload = response.json()
            return payload if isinstance(payload, dict) else None
        except Exception:
            return None

    def _search_google_shopping(self, query: str) -> dict | None:
        try:
            response = requests.get(
                SCRAPINGDOG_GOOGLE_SHOPPING_URL,
                params={
                    "api_key": SCRAPINGDOG_API_KEY,
                    "query": query,
                    "country": AMAZON_COUNTRY,
                    "language": GOOGLE_LANGUAGE,
                    "page": 0,
                    "domain": "google.com",
                },
                timeout=20,
            )
            if response.status_code != 200:
                return None
            payload = response.json()
            return payload if isinstance(payload, dict) else None
        except Exception:
            return None

    def _build_product_url(self, asin: str | None) -> str | None:
        if not asin:
            return None
        return f"https://www.amazon.in/dp/{asin}"

    def _optional_text(self, value: object | None) -> str | None:
        if value is None:
            return None
        text = str(value).strip()
        return text or None

    def _fallback_search_offers(self, query: str, reason: str) -> list[PriceOffer]:
        encoded = quote_plus(query)
        return [
            PriceOffer(
                source="amazon-search",
                provider="Amazon",
                title=query,
                product_url=f"https://www.amazon.in/s?k={encoded}&i=stripbooks",
                note=f"{reason} {AMAZON_LOCATION_LABEL}.",
            ),
            PriceOffer(
                source="google-shopping",
                provider="Google Shopping",
                title=query,
                product_url=f"https://www.google.com/search?tbm=shop&q={encoded}",
                note="Open to compare current listings.",
            ),
            PriceOffer(
                source="abebooks",
                provider="AbeBooks",
                title=query,
                product_url=f"https://www.abebooks.com/servlet/SearchResults?kn={encoded}",
                note="Used and collectible copies.",
            ),
        ]

    def _extract_products(self, payload: object) -> list[dict]:
        if not isinstance(payload, dict):
            return []

        candidate_keys = (
            "products",
            "product_results",
            "results",
            "organic_results",
            "shopping_results",
            "search_results",
        )
        for key in candidate_keys:
            value = payload.get(key)
            if isinstance(value, list):
                return [
                    item
                    for item in value
                    if isinstance(item, dict)
                    and self._optional_text(item.get("type")) in (None, "search_product")
                ]
        return []

    def _extract_primary_offer(self, payload: object) -> dict | None:
        if not isinstance(payload, dict):
            return None
        offers = payload.get("offers")
        if not isinstance(offers, list):
            return None
        for offer in offers:
            if isinstance(offer, dict) and offer.get("buybox_winner"):
                return offer
        for offer in offers:
            if isinstance(offer, dict):
                return offer
        return None

    def _extract_google_shopping_offers(
        self, payload: object, limit: int
    ) -> list[PriceOffer]:
        if not isinstance(payload, dict):
            return []

        results = payload.get("shopping_results")
        if not isinstance(results, list):
            return []

        offers: list[PriceOffer] = []
        for result in results:
            if not isinstance(result, dict):
                continue
            title = self._optional_text(result.get("title"))
            link = self._optional_text(
                result.get("product_url")
                or result.get("product_link")
                or result.get("link")
                or result.get("scrapingdog_immersive_product_link")
            )
            if not title or not link:
                continue
            offers.append(
                PriceOffer(
                    source="google-shopping",
                    provider=self._optional_text(result.get("source"))
                    or "Google Shopping",
                    title=title,
                    price=self._optional_text(result.get("price")),
                    product_url=link,
                    image_url=self._normalize_image_url(
                        self._optional_text(result.get("thumbnail"))
                    ),
                    note=self._build_google_shopping_note(result),
                )
            )
            if len(offers) >= limit:
                break
        return offers

    def _extract_price(self, offer: object) -> str | None:
        if not isinstance(offer, dict):
            return None
        price = offer.get("price")
        if isinstance(price, dict):
            return self._optional_text(price.get("raw"))
        return None

    def _extract_currency(self, offer: object) -> str | None:
        if not isinstance(offer, dict):
            return None
        price = offer.get("price")
        if not isinstance(price, dict):
            return None
        return self._optional_text(price.get("symbol")) or self._optional_text(
            price.get("currency")
        )

    def _build_offer_note(self, product: dict, offer: object) -> str | None:
        note_parts: list[str] = []
        if isinstance(offer, dict):
            delivery = offer.get("delivery")
            if isinstance(delivery, dict):
                delivery_note = self._optional_text(delivery.get("comments")) or self._optional_text(
                    delivery.get("date")
                )
                if delivery_note:
                    note_parts.append(delivery_note)
        fallback_note = self._optional_text(
            product.get("delivery")
            or product.get("delivery_info")
            or product.get("shipping")
            or product.get("badge")
        )
        if fallback_note and fallback_note not in note_parts:
            note_parts.append(fallback_note)
        if AMAZON_LOCATION_LABEL not in " ".join(note_parts):
            note_parts.append(AMAZON_LOCATION_LABEL)
        return " | ".join(part for part in note_parts if part)

    def _build_google_shopping_note(self, result: dict) -> str | None:
        note_parts: list[str] = []
        delivery = self._optional_text(result.get("delivery"))
        if delivery:
            note_parts.append(delivery)
        extensions = result.get("extensions")
        if isinstance(extensions, list):
            note_parts.extend(
                extension
                for extension in (
                    self._optional_text(extension) for extension in extensions[:2]
                )
                if extension
            )
        if AMAZON_LOCATION_LABEL not in " ".join(note_parts):
            note_parts.append(AMAZON_LOCATION_LABEL)
        return " | ".join(note_parts) if note_parts else None

    def _merge_offers(
        self, amazon_offers: list[PriceOffer], google_offers: list[PriceOffer], limit: int
    ) -> list[PriceOffer]:
        combined: list[PriceOffer] = []
        seen_keys: set[str] = set()

        for offer in amazon_offers + google_offers:
            key = (offer.product_url or offer.asin or offer.title).strip().lower()
            if not key or key in seen_keys:
                continue
            combined.append(offer)
            seen_keys.add(key)
            if len(combined) >= limit:
                break
        return combined

    def _normalize_image_url(self, url: str | None) -> str | None:
        if not url or url.startswith("data:"):
            return None
        return url

    def _rank_products(self, query: str, products: list[dict]) -> list[dict]:
        query_tokens = self._tokenize(query)

        def score(product: dict) -> tuple[int, int, int]:
            title = self._optional_text(product.get("title")) or ""
            title_lc = title.lower()
            title_tokens = self._tokenize(title)
            overlap = len(query_tokens & title_tokens)
            missing = len(query_tokens - title_tokens)
            penalty = 0
            if query_tokens and not ("set" in query_tokens or "series" in query_tokens):
                penalty += sum(1 for hint in BOX_SET_HINTS if hint in title_lc)
            return (overlap, -penalty, -missing)

        return sorted(products, key=score, reverse=True)

    def _tokenize(self, text: str) -> set[str]:
        tokens = {
            token
            for token in re.findall(r"[a-z0-9]+", text.lower())
            if len(token) > 1 and token not in {"by", "the", "and", "for", "set", "book"}
        }
        return tokens


price_service = PriceService()
