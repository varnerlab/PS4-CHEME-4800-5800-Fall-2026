"""Validate reference and starter packages in disposable directories.

Run with the required Julia packages available:
    python3 instructor/validate.py
Logs are retained in instructor/validation-output. The working-tree starter
and reference solution are never replaced. No files are published.
"""
from pathlib import Path
import os
import re
import shutil
import subprocess
import tempfile
from build_student_package import ROOT, student_files

LOGS = ROOT / "instructor/validation-output"
LOGS.mkdir(exist_ok=True)
REPORTS = {"accuracy.csv", "confusion-table.csv", "confusion-table.png", "digit-weights.png",
           "misread-digits.png", "logbook-page.csv", "logbook-page.png", "logbook-summary.csv"}

def stage(reference=False):
    """Copy the student package into a new temporary directory and return its Path.

    If reference is True, replace only that copy's Compute.jl and responses.md
    with the reference files. Keep the temporary directory for inspection and
    leave the working tree unchanged. Copying errors propagate to the caller.
    """
    target = Path(tempfile.mkdtemp(prefix="ps4-validation-"))
    for path in student_files():
        dest = target / path.relative_to(ROOT)
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, dest)
    if reference:
        for name in ("src/Compute.jl", "responses.md"):
            shutil.copy2(ROOT / "solution" / name, target / name)
    return target

def run(target, label, expected, script="check_submission.jl", args=(), cwd=None, project="."):
    """Run a Julia script and return its combined stdout/stderr as text.

    target is the staged assignment Path, label names its log, expected is the
    required process exit code, and script is relative to target unless absolute.
    Append args after the script filename (for example, ("--solution",)). Run in
    cwd (default target) with --project=project. Inherit the current environment,
    write LOGS/<label>.log, and raise AssertionError if the exit code differs.
    """
    result = subprocess.run(["julia", "--startup-file=no", f"--project={project}", str(script), *args],
                            cwd=cwd or target, text=True, stdout=subprocess.PIPE,
                            stderr=subprocess.STDOUT, env=os.environ.copy())
    (LOGS / (label + ".log")).write_text(result.stdout)
    assert result.returncode == expected, f"{label}: exit {result.returncode}; see its log"
    print(f"{label}: expected exit {expected}", flush=True)
    return result.stdout

# The report list must agree in the reporting module, the checker, and this script.
names = lambda text: set(re.findall(r'"([a-z-]+\.(?:csv|png))"', text))
reporting = (ROOT / "src/Reporting.jl").read_text()
assert names(reporting.split("const REPORT_FILES", 1)[1].split(");", 1)[0]) == REPORTS
checker = (ROOT / "check_submission.jl").read_text()
assert names(checker.split("Remove earlier reports", 1)[1].split("end", 1)[0]) == REPORTS

run(ROOT, "trainer-checks", 0, script="instructor/test_trainers.jl")
# The direct instructor mode must preserve student files and their output files.
protected = [ROOT / "src/Compute.jl", ROOT / "responses.md", ROOT / "MANIFEST.txt"]
protected += [p for p in (ROOT / "outputs").rglob("*") if p.is_file()]
before = {p: p.read_bytes() if p.exists() else None for p in protected}
output = run(ROOT, "solution-flag-checker", 0, args=("--solution",))
assert output.count("24     24") == 2, output
assert "REFERENCE SOLUTION PASSED" in output and "READY TO PACKAGE" not in output
manifest = (ROOT / "solution/MANIFEST.txt").read_text()
assert "solution/src/Compute.jl" in manifest and "solution/responses.md" in manifest
output = run(ROOT, "solution-flag-part-1", 0, script="testme_part_1.jl", args=("--solution",))
assert "24     24" in output
output = run(ROOT, "solution-flag-driver", 0, script="runlogbook.jl", args=("--solution",))
assert "Training logistic regression" in output and "done in" in output
assert {p.name for p in (ROOT / "solution/outputs").iterdir()} >= REPORTS
assert all((p.read_bytes() if p.exists() else None) == content for p, content in before.items())

reference = stage(reference=True)
output = run(reference, "reference-checker", 0)
assert output.count("24     24") == 2, output
assert "READY TO PACKAGE" in output
# The checker clears exactly these reports, so the list must match what it writes.
assert {p.name for p in (reference / "outputs").iterdir()} == REPORTS
preview = ROOT / "instructor/reference"
preview.mkdir(exist_ok=True)
for path in (reference / "outputs").iterdir():
    shutil.copy2(path, preview / path.name)
accuracy = (reference / "outputs/accuracy.csv").read_bytes()

# Windows line endings and a byte-order mark must not hide the answers.
answers = (reference / "responses.md").read_text()
(reference / "responses.md").write_bytes(b"\xef\xbb\xbf" + answers.replace("\n", "\r\n").encode())
output = run(reference, "crlf-responses", 0)
assert "READY TO PACKAGE" in output
(reference / "responses.md").write_text(answers)

# The driver must work when started from another folder.
shutil.rmtree(reference / "outputs")
output = run(reference, "other-directory-driver", 0, script=reference / "runlogbook.jl",
             cwd=reference.parent, project=str(reference))
assert (reference / "outputs/accuracy.csv").read_bytes() == accuracy

starter = stage()
for script in ("check_submission.jl", "testme_part_1.jl", "runlogbook.jl"):
    output = run(starter, "missing-solution-" + script, 1, script=script, args=("--solution",))
    assert "--solution requires" in output
assert not (starter / "outputs").exists() and not (starter / "MANIFEST.txt").exists()
output = run(starter, "starter-checker", 1)
assert "READY TO PACKAGE" not in output
assert all(f"Question {i} still has a placeholder" in output for i in (1, 2, 3))
assert (starter / "MANIFEST.txt").is_file()
assert not (starter / "outputs").exists()
# An unfinished driver run must stop with a short message naming the function.
output = run(starter, "unfinished-driver", 1, script="runlogbook.jl")
assert "Complete digit_labels" in output and len(output.splitlines()) < 40, output
assert not (starter / "outputs").exists() or not any((starter / "outputs").iterdir())

# A partially completed submission should keep its Part 1 credit.
partial = stage()
source = (ROOT / "solution/src/Compute.jl").read_text()
start = source.index("function predict_digits")
stop = source.index("# ----------------------------------------------------------------------------\n# Part 2b")
source = (source[:start] + 'function predict_digits(W::AbstractMatrix{<:Real}, X::AbstractMatrix{<:Real})\n'
          '    error("unfinished predictions");\nend\n\n\n' + source[stop:])
(partial / "src/Compute.jl").write_text(source)
# Earlier reports must not survive a failing run; other files in outputs/ stay.
(partial / "outputs").mkdir()
for name in ("confusion-table.png", "logbook-summary.csv", "notes.txt"):
    (partial / "outputs" / name).write_text("stale")
output = run(partial, "partial-checker", 1)
assert [p.name for p in (partial / "outputs").iterdir()] == ["notes.txt"]
assert "testme_part_1.jl: all tests passed" in output
assert "testme_part_2.jl: some tests failed" in output

# Load errors must remain distinct from test failures and still yield a manifest.
syntax = stage()
(syntax / "src/Compute.jl").write_text("function unfinished(\n")
output = run(syntax, "syntax-error-checker", 1)
assert "tests could not run" in output
assert (syntax / "MANIFEST.txt").is_file()

# Correct code with missing writing must not be reported ready.
(reference / "responses.md").unlink()
output = run(reference, "missing-responses-checker", 1)
assert "responses.md is missing" in output and "READY TO PACKAGE" not in output

# A rendering failure must not erase passing test results or claim readiness.
shutil.copy2(ROOT / "solution/responses.md", reference / "responses.md")
(reference / "src/Reporting.jl").write_text('error("Deliberate reporting failure")\n')
output = run(reference, "reporting-error-checker", 1)
assert "OUTPUT GENERATION NEEDS ATTENTION" in output
assert "testme_part_1.jl: all tests passed" in output
assert "READY TO PACKAGE" not in output
print("Validated the reference and starter, partial work, other folders, line endings, syntax errors, "
      "missing responses, and reporting failures.")
