import os

# The adapter needs Transformers, but these tests do not use TensorFlow. Some
# developer machines have an unrelated, incompatible TensorFlow installation.
os.environ.setdefault("USE_TF", "0")

from crcbenchmark.models import pipeline_adapter


def test_reasoning_switch_and_generation_options_reach_pipeline(monkeypatch, tmp_path):
    calls = []

    class FakePipeline:
        def __call__(self, **kwargs):
            calls.append(kwargs)
            return [{"generated_text": "A"}]

    monkeypatch.setattr(pipeline_adapter, "pipeline", lambda *args, **kwargs: FakePipeline())
    model = pipeline_adapter.PipelineVLM("unused", enable_thinking=False)
    assert model.generate(tmp_path / "image.png", "Choose A or B") == "A"
    assert calls[0]["enable_thinking"] is False
    assert calls[0]["generate_kwargs"] == {"max_new_tokens": 96, "do_sample": False}
    assert calls[0]["text"][0]["content"][1]["text"] == "Choose A or B"


def test_other_models_receive_no_reasoning_switch(monkeypatch, tmp_path):
    calls = []

    def fake_pipeline(*args, **kwargs):
        def generate(**call_kwargs):
            calls.append(call_kwargs)
            return [{"generated_text": "ABSENT"}]

        return generate

    monkeypatch.setattr(pipeline_adapter, "pipeline", fake_pipeline)
    model = pipeline_adapter.PipelineVLM("unused")
    assert model.generate(tmp_path / "image.png", "Presence?", max_new_tokens=24) == "ABSENT"
    assert "enable_thinking" not in calls[0]
    assert calls[0]["generate_kwargs"]["max_new_tokens"] == 24


def test_text_only_control_omits_image_instead_of_using_dummy_path(monkeypatch):
    calls = []
    def fake_pipeline(*args, **kwargs):
        def generate(**call_kwargs):
            calls.append(call_kwargs)
            return [{"generated_text": '["A"]'}]
        return generate
    monkeypatch.setattr(pipeline_adapter, "pipeline", fake_pipeline)
    model = pipeline_adapter.PipelineVLM("unused", enable_thinking=False)
    assert model.generate(None, "Same frozen prompt") == '["A"]'
    assert calls[0]["text"][0]["content"] == [{"type": "text", "text": "Same frozen prompt"}]
    assert calls[0]["generate_kwargs"] == {"max_new_tokens": 96, "do_sample": False}
