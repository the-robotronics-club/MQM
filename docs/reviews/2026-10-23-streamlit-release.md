# Streamlit campus dashboard release

The campus dining layout, mess selection, sample menus, demo student profiles,
and light theme are included. The free dashboard image copies only the theme
configuration, not Streamlit secrets. Hosted navigation offers live cameras and
comparison; upload/phone modes stay excluded from the lightweight hosted image.

Offline cameras and failed API requests no longer display simulated readings or
invented fresh timestamps. Real distance snapshots retain people, queue, sitting,
and other-standing counts. Capacity appears only when supplied by calibrated
metrics. Wait calculations are explicitly illustrative, based on an uncalibrated
1.3 minutes/person assumption. Menus and dining hall details are sample content.

Comparison buttons update navigation through callbacks, avoiding mutation after
widget creation. Demo profile selection selects its assigned hall. Unknown
student IDs are rejected; the public demo profiles are clearly labelled and do
not establish institute identity. Meal times use Asia/Kolkata.

Validation: 157 tests passed, one opt-in live-stack test skipped, Ruff and MyPy
passed. The full suite ran in the isolated validation copy with committed sample
fixtures; the user's two existing sample deletions remain untouched. AppTest
covers all four snapshot categories, offline/stale comparison behavior,
comparison navigation, API failure reporting, and demo profile selection.

Deployment is requested but not yet verified for this release. Browser/computer
control was stopped at the user's request. Render integration/API authentication
is not currently available in this session, and service auto-deploy is disabled.
The source commit must be deployed to the existing free dashboard service
srv-dbctjqdg1s2s73egdes0 before claiming the public UI is updated. The API service
and model checkpoints are unchanged. Physical Pi uploads remain unverified.
