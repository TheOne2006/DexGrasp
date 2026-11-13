"""Checkpoint identities, including explicit legacy verification for audited migration."""
import ast,hashlib,json
from pathlib import Path
from dexonomy.acceptance_audit import serial

def script_dependencies(driver, scripts):
    """Hash the static sibling-import closure without importing worker/GPU code.

    Includes imports inside functions and transitive helpers. Dynamic imports
    are not a supported way to add production helpers. Missing mandatory files
    fail closed; deleting a discovered helper also changes the dependency map.
    """
    scripts = Path(scripts)
    pending = [Path(driver), scripts / 'night_checkpoint_identity.py']
    paths = {}
    while pending:
        path = pending.pop()
        if path.name in paths:
            continue
        source = path.read_bytes()
        paths[path.name] = hashlib.sha256(source).hexdigest()
        for node in ast.walk(ast.parse(source, filename=str(path))):
            if isinstance(node, ast.Import):
                modules = [alias.name for alias in node.names]
            elif isinstance(node, ast.ImportFrom) and node.level == 0:
                modules = [node.module or '']
            else:
                continue
            for module in modules:
                # Production helpers are flat sibling modules, not packages.
                candidate = scripts / (module.split('.')[0] + '.py')
                if candidate.is_file():
                    pending.append(candidate)
    paths.pop(Path(driver).name, None)  # driver has its own identity field
    return paths

def token(config,driver,sources=None,scripts=None,legacy=False):
    driver=Path(driver);scripts=Path(scripts or driver.parent)
    identity={k:v for k,v in config.items() if k not in ('stop_after_stage','stop_after_jobs','case_limit_s','hard_deadline_epoch','stop_new_init_epoch')}
    identity['driver_sha256']=hashlib.sha256(driver.read_bytes()).hexdigest()
    names=['night_schedule.py','night_templates.py','night_inventory.py']
    if not legacy:names.append('night_checkpoint_identity.py')
    if config.get('capacity_policy'):names.append('night_capacity_policy.py')
    identity['driver_dependencies']={n:hashlib.sha256((scripts/n).read_bytes()).hexdigest() for n in names}
    if not legacy:
        identity['driver_dependencies'].update(script_dependencies(driver, scripts))
    identity['runtime_source']=sources if sources is not None else {str(f.relative_to(config['source'])):hashlib.sha256(f.read_bytes()).hexdigest() for f in (Path(config['source'])/'dexonomy').rglob('*.py')}
    return hashlib.sha256(json.dumps(identity,sort_keys=True,default=serial).encode()).hexdigest()
