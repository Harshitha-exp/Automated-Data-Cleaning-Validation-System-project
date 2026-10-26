# How we work in this repo

## Branches
- `main` only ever holds reviewed code with passing CI. Nobody commits to it directly.
- One short-lived branch per task, named `<module>/<what>`:
  `module1/metadata-engine`, `module2/knn-imputer`, `module3/isolation-forest`.

## Day to day
1. `git checkout main && git pull`
2. `git checkout -b module2/knn-imputer`
3. Work in small commits under your own name, with clear messages.
4. `git push -u origin module2/knn-imputer`, then open a pull request into `main`.
5. Fill in the pull request template. One teammate reviews; CI must be green.
6. Squash-merge, delete the branch, everyone runs `git pull` on `main`.

## Shared contracts
The files in `schemas/` are the agreements between modules:
Module 1 writes `profiling_report.json`, Module 2 reads it and writes
`cleaned_data.csv` + `cleaning_log.json`, Module 3 writes `validation_report.json`.
Changing one of these changes someone else's input, so tag the other module
owners on that pull request.

## Weekly
The rotating Scrum Master runs the weekly meeting from the GitHub Issues /
Projects board (To do, In progress, In review, Done) and checks that open
pull requests aren't stuck.

## Never commit
Real datasets, generated reports, secrets, or `__pycache__`. See `.gitignore`.
Small demo samples go in `sample_data/`.
