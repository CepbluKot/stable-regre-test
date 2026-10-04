from pathlib import Path

from streamlit.testing.v1 import AppTest

APP = Path(__file__).resolve().parents[1] / "app.py"


def test_changed_model_hides_old_results_until_run():
    app = AppTest.from_file(str(APP), default_timeout=20).run()
    app.button[0].click().run()
    assert any(m.label == "Observed points" for m in app.metric)
    assert any("expected ± 1.96" in x.value for x in app.get("caption"))
    next(box for box in app.selectbox if box.label == "Model").set_value(
        "seasonal"
    ).run()
    assert not any(m.label == "Observed points" for m in app.metric)
    assert any("settings changed" in item.value.lower() for item in app.warning)


def test_labeled_upload_and_invalid_upload_replace_results():
    fixture = Path(__file__).parent / "fixtures/labeled-shift.csv"
    app = AppTest.from_file(str(APP), default_timeout=20).run()
    app.file_uploader[0].set_value(
        (fixture.name, fixture.read_bytes(), "text/csv")
    ).run()
    app.button[0].click().run()
    assert any(m.label == "Observed points" and m.value == "120" for m in app.metric)
    assert any("Retrospective ranking diagnostics" in x.value for x in app.markdown)
    app.file_uploader[0].set_value(
        ("bad.csv", b"timestamp,value_0\n1,not-a-number\n2,3\n3,4\n", "text/csv")
    ).run()
    app.button[0].click().run()
    assert app.error and not app.metric
