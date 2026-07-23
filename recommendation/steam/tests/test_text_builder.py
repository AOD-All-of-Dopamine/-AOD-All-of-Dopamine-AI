# tests/test_text_builder.py
from src.text_builder import build_semantic_text


def test_with_genres():
    text = build_semantic_text("Dig, fight, explore, build!", ["액션", "어드벤처", "인디"])
    assert text == "Description: Dig, fight, explore, build!\nGenres: 액션, 어드벤처, 인디"


def test_without_genres_no_placeholder():
    text = build_semantic_text("A quiet puzzle game.", [])
    assert text == "Description: A quiet puzzle game."
    assert "Unknown" not in text
    assert "Genres" not in text
