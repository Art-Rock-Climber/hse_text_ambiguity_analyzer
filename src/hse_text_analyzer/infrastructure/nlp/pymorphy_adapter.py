from __future__ import annotations

from pymorphy3 import MorphAnalyzer

from hse_text_analyzer.domain.models.linguistic import MorphologicalFeatures


class PymorphyMorphologyAdapter:
    def __init__(self) -> None:
        self._analyzer = MorphAnalyzer()

    def analyze(self, token: str) -> MorphologicalFeatures:
        if not token or not token.strip():
            return MorphologicalFeatures()

        parsed = self._analyzer.parse(token)[0]
        tag = parsed.tag
        return MorphologicalFeatures(
            lemma=parsed.normal_form,
            pos=tag.POS or None,
            tag=str(tag) if tag else None,
            grammemes=list(tag.grammemes) if tag else [],
            normal_form=parsed.normal_form,
            is_known=parsed.is_known,
            gender=getattr(tag, "gender", None),
            number=getattr(tag, "number", None),
            case=getattr(tag, "case", None),
            tense=getattr(tag, "tense", None),
            aspect=getattr(tag, "aspect", None),
            person=getattr(tag, "person", None),
            mood=getattr(tag, "mood", None),
            voice=getattr(tag, "voice", None),
        )
