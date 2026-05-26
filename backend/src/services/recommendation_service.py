from __future__ import annotations

from collections import defaultdict
from datetime import datetime, timezone
import os
import re
from typing import Any

import requests

from src.models import BookRecommendation
from src.services.auth_service import AuthenticatedUser
from src.services.mongo_service import mongo_service


WeightedTerms = defaultdict[str, float]
RecommendationProfile = dict[str, WeightedTerms]


class RecommendationService:
    def __init__(self) -> None:
        self.google_books_api_key = os.getenv("GOOGLE_BOOKS_API_KEY", "").strip()
        self.stop_words = {
            "a",
            "an",
            "and",
            "at",
            "by",
            "for",
            "from",
            "in",
            "of",
            "on",
            "or",
            "series",
            "the",
            "to",
            "with",
        }

    def list_recommendations(
        self, user: AuthenticatedUser, limit: int = 12
    ) -> list[BookRecommendation]:
        books_collection = mongo_service.get_books_collection()
        search_collection = mongo_service.get_search_history_collection()

        library_docs = list(
            books_collection.find(
                {"auth0UserId": user.auth0_user_id},
                {"_id": 0, "auth0UserId": 0},
            )
        )
        search_docs = list(
            search_collection.find(
                {"auth0UserId": user.auth0_user_id},
                {"_id": 0, "auth0UserId": 0},
            )
            .sort("createdAt", -1)
            .limit(40)
        )

        if not library_docs and not search_docs:
            return []

        profile = self._build_profile(library_docs, search_docs)
        candidate_queries = self._build_candidate_queries(profile)
        if not candidate_queries:
            return []

        owned_ids = {str(doc.get("id", "")) for doc in library_docs}
        scored_candidates: dict[str, BookRecommendation] = {}
        for query in candidate_queries[:6]:
            for candidate in self._search_candidates(query, 10):
                if candidate.id in owned_ids:
                    continue
                enriched = self._score_candidate(candidate, profile, query)
                current = scored_candidates.get(enriched.id)
                if current is None or enriched.score > current.score:
                    scored_candidates[enriched.id] = enriched

        ranked = sorted(
            scored_candidates.values(),
            key=lambda item: item.score,
            reverse=True,
        )
        return self._diversify(ranked, limit)

    def _build_profile(
        self, library_docs: list[dict[str, Any]], search_docs: list[dict[str, Any]]
    ) -> RecommendationProfile:
        profile = {
            "authors": defaultdict(float),
            "subjects": defaultdict(float),
            "keywords": defaultdict(float),
            "queries": defaultdict(float),
        }

        for doc in library_docs:
            weight = self._kept_book_weight(doc.get("createdAt"))
            for author in doc.get("authors") or []:
                profile["authors"][self._normalize_text(author)] += weight * 1.4
            for subject in doc.get("subjects") or []:
                profile["subjects"][self._normalize_text(subject)] += weight * 1.2
            for token in self._tokenize(doc.get("title", "")):
                profile["keywords"][token] += weight * 0.85

        for index, doc in enumerate(search_docs):
            weight = max(0.45, 1.8 - (index * 0.08))
            for token in self._tokenize(doc.get("normalizedQuery") or doc.get("query", "")):
                profile["queries"][token] += weight
            for author in doc.get("selectedAuthors") or []:
                profile["authors"][self._normalize_text(author)] += weight * 1.1
            for subject in doc.get("selectedSubjects") or []:
                profile["subjects"][self._normalize_text(subject)] += weight
            for token in self._tokenize(doc.get("selectedTitle", "")):
                profile["keywords"][token] += weight * 0.9

        return profile

    def _build_candidate_queries(self, profile: RecommendationProfile) -> list[str]:
        queries: list[str] = []
        queries.extend(self._top_terms(profile["authors"], 3))
        queries.extend(self._top_terms(profile["subjects"], 4))
        query_tokens = self._top_terms(profile["queries"], 4)
        if query_tokens:
            queries.append(" ".join(query_tokens[:2]))
            queries.extend(query_tokens)

        keyword_tokens = self._top_terms(profile["keywords"], 4)
        if keyword_tokens:
            queries.append(" ".join(keyword_tokens[:2]))

        deduped: list[str] = []
        seen: set[str] = set()
        for query in queries:
            normalized = self._normalize_text(query)
            if not normalized or normalized in seen:
                continue
            seen.add(normalized)
            deduped.append(query)
        return deduped

    def _search_candidates(self, query: str, limit: int) -> list[BookRecommendation]:
        open_library_candidates = self._search_open_library(query, limit)
        google_candidates = self._search_google_books(query, limit)
        merged: dict[str, BookRecommendation] = {}
        for candidate in [*open_library_candidates, *google_candidates]:
            dedupe_key = self._dedupe_key(candidate)
            current = merged.get(dedupe_key)
            if current is None or self._candidate_richness(candidate) > self._candidate_richness(current):
                merged[dedupe_key] = candidate
        return list(merged.values())

    def _search_open_library(self, query: str, limit: int) -> list[BookRecommendation]:
        try:
            response = requests.get(
                "https://openlibrary.org/search.json",
                params={"q": query, "limit": limit},
                timeout=10,
            )
            response.raise_for_status()
        except requests.RequestException:
            return []

        docs = response.json().get("docs", [])
        items: list[BookRecommendation] = []
        for doc in docs:
            book_id = str(doc.get("key") or "").strip()
            title = str(doc.get("title") or "").strip()
            if not book_id or not title:
                continue
            cover_id = doc.get("cover_i")
            items.append(
                BookRecommendation(
                    id=book_id,
                    title=title,
                    authors=[str(author).strip() for author in doc.get("author_name") or [] if str(author).strip()],
                    cover_url=(
                        f"https://covers.openlibrary.org/b/id/{cover_id}-L.jpg"
                        if cover_id
                        else None
                    ),
                    subjects=[str(subject).strip() for subject in doc.get("subject") or [] if str(subject).strip()][:12],
                    published_date=str(doc.get("first_publish_year")) if doc.get("first_publish_year") else None,
                    isbn=(doc.get("isbn") or [None])[0],
                    source="openlibrary",
                    score=0,
                )
            )
        return items

    def _search_google_books(self, query: str, limit: int) -> list[BookRecommendation]:
        params = {"q": query, "maxResults": min(limit, 20)}
        if self.google_books_api_key:
            params["key"] = self.google_books_api_key

        try:
            response = requests.get(
                "https://www.googleapis.com/books/v1/volumes",
                params=params,
                timeout=10,
            )
            response.raise_for_status()
        except requests.RequestException:
            return []

        items = response.json().get("items", [])
        recommendations: list[BookRecommendation] = []
        for item in items:
            volume_id = str(item.get("id") or "").strip()
            volume_info = item.get("volumeInfo") or {}
            title = str(volume_info.get("title") or "").strip()
            if not volume_id or not title:
                continue
            image_links = volume_info.get("imageLinks") or {}
            identifiers = volume_info.get("industryIdentifiers") or []
            isbn = None
            for identifier in identifiers:
                value = str(identifier.get("identifier") or "").strip()
                if value:
                    isbn = value
                    break
            recommendations.append(
                BookRecommendation(
                    id=f"googlebooks:{volume_id}",
                    title=title,
                    authors=[str(author).strip() for author in volume_info.get("authors") or [] if str(author).strip()],
                    cover_url=image_links.get("thumbnail") or image_links.get("smallThumbnail"),
                    description=volume_info.get("description"),
                    summary=volume_info.get("description"),
                    subjects=[str(subject).strip() for subject in volume_info.get("categories") or [] if str(subject).strip()],
                    published_date=volume_info.get("publishedDate"),
                    page_count=volume_info.get("pageCount"),
                    isbn=isbn,
                    source="googlebooks",
                    score=0,
                )
            )
        return recommendations

    def _score_candidate(
        self,
        candidate: BookRecommendation,
        profile: RecommendationProfile,
        source_query: str,
    ) -> BookRecommendation:
        author_terms = [self._normalize_text(author) for author in candidate.authors]
        subject_terms = [self._normalize_text(subject) for subject in candidate.subjects or []]
        candidate_tokens = set(self._tokenize(candidate.title))
        candidate_tokens.update(self._tokenize(candidate.description or ""))
        candidate_tokens.update(self._tokenize(candidate.summary or ""))
        candidate_tokens.update(self._tokenize(" ".join(candidate.subjects or [])))

        author_score, matched_authors = self._weighted_overlap(author_terms, profile["authors"])
        subject_score, matched_subjects = self._weighted_overlap(subject_terms, profile["subjects"])
        query_score, matched_queries = self._weighted_overlap(
            list(candidate_tokens), profile["queries"]
        )
        keyword_score, _ = self._weighted_overlap(list(candidate_tokens), profile["keywords"])
        source_query_tokens = set(self._tokenize(source_query))
        source_match_boost = 0.2 if source_query_tokens.intersection(candidate_tokens) else 0

        score = (
            (author_score * 0.38)
            + (subject_score * 0.32)
            + (query_score * 0.18)
            + (keyword_score * 0.12)
            + source_match_boost
        )

        reason = self._build_reason(matched_authors, matched_subjects, matched_queries)
        return candidate.model_copy(
            update={
                "score": round(score, 4),
                "reason": reason,
                "matched_authors": matched_authors[:3],
                "matched_subjects": matched_subjects[:3],
                "matched_queries": matched_queries[:4],
            }
        )

    def _weighted_overlap(
        self, terms: list[str], profile_weights: dict[str, float]
    ) -> tuple[float, list[str]]:
        matched: list[tuple[str, float]] = []
        total = 0.0
        for term in terms:
            normalized = self._normalize_text(term)
            if not normalized:
                continue
            weight = profile_weights.get(normalized)
            if weight:
                matched.append((term, weight))
                total += weight
        matched.sort(key=lambda item: item[1], reverse=True)
        return total, [term for term, _ in matched]

    def _build_reason(
        self,
        matched_authors: list[str],
        matched_subjects: list[str],
        matched_queries: list[str],
    ) -> str:
        if matched_authors:
            return f"Because you keep books by {matched_authors[0]}"
        if matched_subjects:
            return f"Because you often keep {matched_subjects[0]} books"
        if matched_queries:
            return f"Because you recently searched for {matched_queries[0]}"
        return "Picked from your library themes and recent searches"

    def _diversify(
        self, ranked: list[BookRecommendation], limit: int
    ) -> list[BookRecommendation]:
        selected: list[BookRecommendation] = []
        author_counts: dict[str, int] = defaultdict(int)
        for candidate in ranked:
            primary_author = self._normalize_text(candidate.authors[0]) if candidate.authors else ""
            if primary_author and author_counts[primary_author] >= 2:
                continue
            selected.append(candidate)
            if primary_author:
                author_counts[primary_author] += 1
            if len(selected) >= limit:
                break
        return selected

    def _kept_book_weight(self, created_at: str | None) -> float:
        if not created_at:
            return 1.0
        try:
            created = datetime.fromisoformat(created_at.replace("Z", "+00:00"))
        except ValueError:
            return 1.0
        age_days = max((datetime.now(timezone.utc) - created).days, 0)
        return min(1.0 + (age_days / 90), 4.0)

    def _candidate_richness(self, candidate: BookRecommendation) -> int:
        return sum(
            1
            for value in [
                candidate.cover_url,
                candidate.description,
                candidate.summary,
                candidate.subjects,
                candidate.published_date,
                candidate.isbn,
            ]
            if value
        )

    def _dedupe_key(self, candidate: BookRecommendation) -> str:
        author = self._normalize_text(candidate.authors[0]) if candidate.authors else ""
        return f"{self._normalize_text(candidate.title)}::{author}"

    def _top_terms(self, weighted_terms: dict[str, float], limit: int) -> list[str]:
        return [
            term
            for term, _ in sorted(
                weighted_terms.items(), key=lambda item: item[1], reverse=True
            )[:limit]
        ]

    def _tokenize(self, value: str) -> list[str]:
        tokens = re.findall(r"[a-z0-9']+", value.lower())
        return [token for token in tokens if len(token) > 2 and token not in self.stop_words]

    def _normalize_text(self, value: str | None) -> str:
        if not value:
            return ""
        normalized = re.sub(r"\s+", " ", value.strip().lower())
        return normalized


recommendation_service = RecommendationService()
