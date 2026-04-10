from pathlib import Path


def test_ci_uploads_failure_bundle_on_failure():
    workflow = Path('.github/workflows/ci.yml').read_text(encoding='utf-8')

    assert 'Build failure bundle (minimal)' in workflow
    assert 'Build failure bundle (full)' in workflow
    assert 'Upload failure bundle (minimal)' in workflow
    assert 'Upload failure bundle (full)' in workflow
    assert 'if: failure()' in workflow
    assert 'path: failure_bundle/' in workflow
