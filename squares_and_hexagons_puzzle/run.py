import click
import datetime
import pathlib
import time
from typing import Collection, Never, Tuple

import llm
import logging
import rich.logging as rich_logging
import rich.progress as rich_progress


_logger = logging.getLogger(__name__)


_forbidden_models = set({
    # Missing access to model
    "gpt-4-32k",
    "gpt-4.5-preview-2025-02-27",
    "gpt-4.5-preview",
    # aliases
    "gpt-3.5-turbo-16k",                    # gpt-3.5-turbo
    "gpt-3.5-turbo-instruct",               # gpt-3.5-turbo
    "gpt-4-turbo-2024-04-09",               # gpt-4-turbo
    "o1-2024-12-17",                        # o1
    "gpt-5-2025-08-07",                     # gpt-5
    "gpt-5-mini-2025-08-07",                # gpt-5-mini
    "gpt-5-nano-2025-08-07",                # gpt-5-nano
    "anthropic/claude-3-7-sonnet-20250219", # anthropic/claude-3-7-sonnet-latest
    "mistral/ministral-3b-2410",            # mistral/ministral-3b-latest
    "mistral/ministral-8b-2410",            # mistral/ministral-8b-latest
    "mistral/open-mistral-nemo-2407",       # mistral/open-mistral-nemo
    "mistral/mistral-small-2312",           # mistral/mistral-small-latest
    "mistral/mistral-small-2402",           # mistral/mistral-small-latest
    "mistral/mistral-small-2409",           # mistral/mistral-small-latest
    "mistral/mistral-small",                # mistral/mistral-small-latest
    "mistral/mistral-medium-2312",          # mistral/mistral-medium-latest
    "mistral/mistral-medium",               # mistral/mistral-medium-latest
    "mistral/mistral-large-2402",           # mistral/mistral-large-latest
    "mistral/mistral-large-2407",           # mistral/mistral-large-latest
    "mistral/mistral-large-2411",           # mistral/mistral-large-latest
    "mistral/pixtral-large-2411",           # mistral/pixtral-large-latest
    "mistral/codestral-2405",               # mistral/codestral-latest
    "mistral/codestral-2501",               # mistral/codestral-latest
    "mistral/codestral-2412",               # mistral/codestral-latest
    "mistral/codestral-2411-rc5",           # mistral/codestral-latest
    "gemini/gemini-2.5-flash-lite",         # gemini/gemini-flash-lite-latest
    "grok-4-fast-reasoning-latest",         # grok-4-fast
    "grok-3-fast-latest",                   # grok-3-latest
    "grok-3-mini-fast-latest",              # grok-3-mini-latest
    # deprecated (+ aliases)
    "o1-preview",
    "o1-mini",
    "anthropic/claude-3-opus-20240229",
    "anthropic/claude-3-opus-latest",
    "anthropic/claude-3-sonnet-20240229",
    "anthropic/claude-3-5-sonnet-20240620",
    "anthropic/claude-3-5-sonnet-20241022",
    "anthropic/claude-3-5-sonnet-latest",
    "anthropic/claude-3-5-haiku-latest",
    "sonar-reasoning",
    "mistral/open-mistral-7b",
    "mistral/mistral-tiny",
    "mistral/mistral-tiny-2312",
    "mistral/mistral-tiny-2407",
    "mistral/mistral-tiny-latest",
    "mistral/open-mixtral-8x7b",
    "mistral/open-mixtral-8x22b",
    "mistral/open-mixtral-8x22b-2404",
    "mistral/open-codestral-mamba",
    "mistral/codestral-mamba-2407",
    "mistral/codestral-mamba-latest",
    "mistral/pixtral-12b-2409",
    "mistral/pixtral-12b",
    "mistral/pixtral-12b-latest",
    "gemini/gemini-2.5-pro-preview-03-25",
    "gemini/gemini-2.5-flash-preview-05-20",
    "gemini/gemini-2.5-pro-preview-05-06",
    "gemini/gemini-2.5-pro-preview-06-05",
    # Not present in the official documentation (deprecated)
    "gemini/gemini-pro",
    "gemini/gemini-1.5-pro-latest",
    "gemini/gemini-1.5-flash-latest",
    "gemini/gemini-1.5-pro-001",
    "gemini/gemini-1.5-flash-001",
    "gemini/gemini-1.5-pro-002",
    "gemini/gemini-1.5-flash-002",
    "gemini/gemini-1.5-flash-8b-latest",
    "gemini/gemini-1.5-flash-8b-001",
    "gemini/gemini-exp-1114",
    "gemini/gemini-exp-1121",
    "gemini/gemini-exp-1206",
    "gemini/learnlm-1.5-pro-experimental",
    "gemini/gemini-2.0-flash-thinking-exp-1219",
    "gemini/gemini-2.0-flash-thinking-exp-01-21",
    "gemini/gemini-2.0-pro-exp-02-05",
    "gemini/gemini-2.5-pro-exp-03-25",
    "gemini/gemini-2.5-flash-preview-04-17",
    # Task not suitable (e.g., models are audio)
    "gpt-4o-audio-preview",
    "gpt-4o-audio-preview-2024-10-01",
    "gpt-4o-audio-preview-2024-12-17",
    "gpt-4o-mini-audio-preview",
    "gpt-4o-mini-audio-preview-2024-12-17",
    "mistral/mistral-moderation-2411",
    "mistral/mistral-moderation-latest",
    "grok-2-vision-latest",
    # Media resolution bug (https://github.com/simonw/llm-gemini/issues/116)
    "gemini/gemma-3-1b-it",
    "gemini/gemma-3-4b-it",
    "gemini/gemma-3-12b-it",
    "gemini/gemma-3-27b-it",
    "gemini/gemma-3n-e4b-it",
    # Other reasons
    "r1-1776",       # DeepSeek model, different provider
    "grok-2-latest", # Old and costly
    "grok-4-latest", # Very slow, and costly ($1.16 for only 4 answers in over 12 hours)
    # already ran
})


@click.command()
@click.option(
    "--log_level",
    default=logging.INFO,
    help="Log verbosity level.",
)
@click.option(
    "-p",
    "--prompts",
    type=click.File("r"),
    help="File(s) with prompt text.",
    multiple=True,
)
@click.option(
    "-m",
    "--models",
    type=str,
    metavar="MODEL",
    help="Models to query.",
    multiple=True,
)
@click.option(
    "-n",
    "--rounds",
    type=int,
    metavar="ROUNDS",
    help="Number of rounds of querying.",
    default=10,
)
def main(
    log_level: int,
    prompts: Collection[click.File],
    models: Collection[str],
    rounds: int,
) -> None:
    """The main entry point."""
    _setup_logging(log_level)
    _logger.info("Prompt files: %s", prompts)
    _logger.info("Models asked for: %s", models)
    _logger.info("Number of rounds: %d", rounds)
    if not prompts:
        _logger.debug("No prompt given, will just list the models..")
        click.echo("Available models:")
        for model in llm.get_models():
            if model.model_id not in _forbidden_models:
                click.echo(f"\t{model.model_id}")
        raise SystemExit
    if models:
        models = [llm.get_model(m) for m in models]
    else:
        models = []
        for model in llm.get_models():
            if model.model_id not in _forbidden_models:
                models.append(model)
    _run_prompts(prompts, models, rounds)


def _run_prompts(
    prompts: Collection[click.File],
    models: Collection[llm.Model],
    rounds: int,
) -> None:
    """Runs all prompts on all models."""
    now = datetime.datetime.today()
    root_folder = pathlib.Path(now.strftime("%y%m%d-%H%M%S"))
    num_model_runs = len(models) * rounds
    num_tasks = len(prompts) * num_model_runs
    with rich_progress.Progress(
        rich_progress.TextColumn("{task.description}"),
        rich_progress.TaskProgressColumn(),
        rich_progress.BarColumn(None),
        rich_progress.MofNCompleteColumn(),
        rich_progress.TextColumn("•"),
        rich_progress.TimeElapsedColumn(),
        rich_progress.TextColumn("/"),
        rich_progress.TimeRemainingColumn(),
    ) as pbar:
        total_task = pbar.add_task("Total run progress", total=num_tasks)
        prompt_task = pbar.add_task("placeholder", total=num_model_runs)
        model_task = pbar.add_task("placeholder", total=rounds)
        score_task = pbar.add_task("Model score on task", total=rounds)
        for prompt in prompts:
            prompt_name = _canonicalize(prompt.name)
            prompt_folder = root_folder / prompt_name
            pbar.reset(prompt_task, description=prompt_name)
            prompt_text, expected_answer = _read_prompt(prompt)
            for model in models:
                model_name = _canonicalize(model.model_id)
                model_folder = prompt_folder / model_name
                model_folder.mkdir(parents=True)
                pbar.reset(model_task, description=model_name)
                pbar.reset(score_task)
                correct = 0
                pbar.reset(score_task)
                for i in range(rounds):
                    out = model_folder / f"{i}"
                    if _get_answer(prompt_text, expected_answer, model, out):
                        correct += 1
                        pbar.update(score_task, advance=1)
                    pbar.update(model_task, advance=1)
                    pbar.update(prompt_task, advance=1)
                    pbar.update(total_task, advance=1)
                _logger.info(
                    f"{correct}/{rounds} for {model_name} on {prompt_name}"
                )
                (model_folder / "score").write_text(f"{correct}\n")


def _get_answer(
    prompt: str,
    expected: int,
    model: llm.Model,
    log: pathlib.Path,
) -> bool:
    """Queries `model` on `prompt`, expecting `expected`, loging to `log`."""
    wait_time = 1
    while True:
        try:
            with log.open('w') as f:
                response = model.prompt(prompt)
                last_number, current_number = 0, 0
                for part in response:
                    f.write(part)
                    for c in part:
                        if c in "0123456789":
                            current_number = 10 * current_number + int(c)
                        elif current_number:
                            last_number, current_number = current_number, 0
                    if current_number:
                        last_number = current_number
                f.write("\n----\n")
                f.write(f"{response.usage()}\n")
                return last_number == expected
        except Exception as e:
            _logger.exception(
                "Model %s failed, wait %d seconds", model.model_id, wait_time
            )
            time.sleep(wait_time)
            wait_time *= 2


def _read_prompt(prompt: click.File) -> Tuple[str, int]:
    """Parses a prompt file, extracting the prompt and the answer."""
    lines = [l.rstrip() for l in prompt.readlines()]
    return "\n".join(lines[:-1]), int(lines[-1])


def _canonicalize(name: str) -> str:
    """Makes `name` be usable as a filename/task."""
    return name.replace("/", "_")


def _setup_logging(level: int = logging.INFO) -> None:
    """Configures the global logging for the entire run."""
    global _logger
    handler = rich_logging.RichHandler(
        show_level=False, show_time=False, show_path=False
    )
    handler.setFormatter(
        logging.Formatter(
            fmt="%(asctime)s.%(msecs)d - %(levelname).1s - %(message)s",
            datefmt="%m/%d/%Y %H:%M:%S",
        )
    )
    _logger.addHandler(handler)
    _logger.setLevel(level)


if __name__ == "__main__":
    main()
