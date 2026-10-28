# Distribution privacy rules

Every PCS Analyzer release must exclude private user information and activity.
This is a release requirement, not an optional cleanup task.

- Build into a fresh folder. Run the application and QA plugins only in a
  separate disposable copy; archive the unrun release folder.
- Include only the reviewed public documentation and example manifests.
  Exclude private projects, local paths, user/computer identifiers, settings,
  recent files, installed plugins, registries, logs, caches and developer files.
- Never ship passwords, access tokens, private keys or account credential files.
- Review measurement metadata. Identity/contact/owner fields must be empty or
  anonymized; preserve FIDs, scientific acquisition values and numerical behavior.
- Review visible screenshot content as well as image metadata. Metadata scanning
  alone does not establish that a screenshot contains no private information.
- Public author/citation details (including the published email, affiliation and
  ORCID) are approved by the project owner. Required copyright/license texts and
  public third-party attribution must remain. This is not a general exemption
  for private contacts, accounts or arbitrary user data.
- Run `check_release_privacy(release_folder, project_root)` from
  `tools/portable_build_support.py` against the actual finished folder. The local
  portable spec invokes this gate after assembly; packaging must stop on failure.
- Inspect embedded Python code and archive resources as well as ordinary files.
  Record coverage and findings locally, with matched private values redacted.
- Verify every archived file against the inspected payload and provide a SHA-256
  checksum. If the payload changes, repeat the affected checks before packaging.

Automated pattern checks complement the allowlists and content review; they do
not prove the absence of every possible personal datum or complete licensing
review. Never describe unperformed checks as passed.
