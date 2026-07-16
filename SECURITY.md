# Security

## Reporting

Report vulnerabilities privately via GitHub's "Report a vulnerability"
(Security tab) or to nicholas.ehsan.roy@gmail.com. Please do not open public
issues for suspected vulnerabilities.

Soundness defects (a wrong verdict) are tracked openly in
[SOUNDNESS.md](SOUNDNESS.md) — but if a soundness defect has security
consequences for your use, report it privately first.

## Supply chain

Releases are published to PyPI exclusively by
[`.github/workflows/release.yml`](.github/workflows/release.yml) via PyPI
Trusted Publishing (OIDC). There are no PyPI API tokens — in CI or anywhere
else. The publish action generates PEP 740 attestations by default, binding
every uploaded wheel and sdist to the exact source commit and workflow run
that built it.

To verify a distribution independently, fetch its provenance from the PyPI
Integrity API:

```
https://pypi.org/integrity/stelling/<version>/<filename>/provenance
```

or use the `pypi-attestations` tool to check a downloaded file against this
repository's identity (`https://github.com/NicholasEhsanRoy/stelling`). See
PyPI's attestation documentation: https://docs.pypi.org/attestations/

What is *not* attested: none of this proves the code is correct — see
[SOUNDNESS.md](SOUNDNESS.md) for what a verdict does and does not claim.
