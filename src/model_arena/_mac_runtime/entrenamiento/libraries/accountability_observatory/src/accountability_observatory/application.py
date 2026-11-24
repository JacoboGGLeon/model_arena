import json
from pathlib import Path
import sys
import yaml
from pydantic import BaseModel


def scaffold(path, *, bundle):
    """Create a monitor project with a portable YAML pointing to the extracted bundle."""
    import os
    dest=Path(path).resolve()
    bundle=Path(bundle).resolve()
    if not (bundle/'selector_observatorio.py').is_file():
        raise ValueError('Se requiere la raíz del bundle')
    dest.mkdir(parents=True,exist_ok=True)
    config=dest/'monitor.yaml'
    with config.open('x') as f:
        yaml.safe_dump({'bundle':os.path.relpath(bundle,dest),'run':None,
                        'search_root':os.path.relpath(bundle.parent,dest)},f)
    return config


class Inspection(BaseModel):
    run: str
    accounts: int = 0
    hpo_ready: bool = False


def inspect_run(run):
    """Read the source manifest and the availability of HPO evidence in a separate process."""
    import agentic_systems as sdk
    def inspect(state: Inspection):
        model=Path(state.run)
        manifest=json.loads((model/'manifest.json').read_text())
        return {'accounts':manifest['accounts'],'hpo_ready':(model.parent/'hpo_f1_f3/precision_modelos.json').is_file()}
    graph=sdk.Graph(Inspection)
    graph.add_node('inspect',inspect)
    graph.add_edge(sdk.START,'inspect')
    graph.add_edge('inspect',sdk.END)
    result=sdk.System('observatory_reader',graph=graph).run({'run':str(run)})
    if not result.ok:raise ValueError(str(result.errors))
    return result.data


def monitor(config, *, run=None, python=None):
    """Launch the web server from the project YAML; return the server process."""
    import importlib.util
    path=Path(config).resolve()
    settings=yaml.safe_load(path.read_text())
    bundle=(path.parent/settings['bundle']).resolve()
    module=importlib.util.spec_from_file_location('observatorio_launcher',bundle/'abrir_observatorio.py')
    launcher=importlib.util.module_from_spec(module)
    module.loader.exec_module(launcher)
    # Legacy launcher imports sibling modules by name during discovery.
    sys.path.insert(0,str(bundle))
    try:
        target=run or settings.get('run')
        if target:
            target=(path.parent/target).resolve()
            if (target/'modelado').is_dir():target=target/'modelado'
        return launcher.launch(target,python=python,search_root=(path.parent/settings['search_root']).resolve())
    finally:
        sys.path.remove(str(bundle))
