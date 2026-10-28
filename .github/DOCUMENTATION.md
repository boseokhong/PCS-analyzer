# Publishing the versioned manual

Pushing to `main` validates the manual with `mkdocs build --strict`, then deploys
the version from `APP_VERSION` in `app_version.py` with mike. For the current
source this publishes `/1.4.0/` and points `/latest/` at it. This does not create
a Git tag or GitHub Release, or upload a portable application.

The original `/1.3.4/` manual and its assets are preserved by the archive guard;
the workflow refuses to rebuild that historical version. Publication uses normal
Git pushes to `gh-pages`, without rewriting its history.

- Pull requests validate the manual without publishing it.
- A published, non-prerelease GitHub Release also publishes the matching manual;
  its tag must agree with `APP_VERSION`.
- Manual `preview` publishes `/dev/` without moving `latest`.
- Manual `archive-only` preserves the existing site without publishing new pages.

The documentation version can be published before the matching application
release. A successful code push alone is not proof of publication: check the
documentation workflow's publish job and the served `versions.json` afterward.
