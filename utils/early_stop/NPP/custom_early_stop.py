from utils.register import register_early_stop
from utils.logger import get_logger

logger = get_logger(__name__)

@register_early_stop("default")
def custom_early_stop(res:dict[str, float], best_res:dict[str, float]) -> tuple[bool, dict[str, float]]:
    flag = True
    if best_res == {}:
        return True, res
    
    for key, value in res.items():
        if key == 'MRR':
            continue
        if key not in best_res:
            raise ValueError(f"Key {key} not in best_res")
        if value <= best_res[key]:
            logger.info(f"Metric {key:>10s} : {value:.5f} <= {best_res[key]:.5f}")
            flag = False
        else:
            logger.info(f"Metric {key:>10s} : {value:.5f}")

    if flag:
        return True, res
    return False, best_res

@register_early_stop("half_improve")
def half_improve_early_stop(res:dict[str, float], best_res:dict[str, float]) -> tuple[bool, dict[str, float]]:
    if best_res == {}:
        return True, res
    unimproved_count = 0

    for key, value in res.items():
        if key == 'MRR':
            continue
        if key not in best_res:
            raise ValueError(f"Key {key} not in best_res")
        if value <= best_res[key]:
            unimproved_count += 1
            logger.info(f"{unimproved_count:3d}th metric {key:>10s} did not improve: {value:.5f} <= {best_res[key]:.5f}")
        else:
            logger.info(f"      metric {key:>10s} improved: {value:.5f}")

    if unimproved_count <= len(res) / 2:
        return True, res
    return False, best_res

@register_early_stop("sum_improve")
def sum_improve_early_stop(res:dict[str, float], best_res:dict[str, float]) -> tuple[bool, dict[str, float]]:
    for key, value in res.items():
        logger.info(f"Metric {key:>10s} : {value:.5f}")
    if best_res == {}:
        return True, res
    new_sum = sum(res.values())
    best_sum = sum(best_res.values())
    if new_sum > best_sum:
        return True, res
    return False, best_res

@register_early_stop("quarter_improve")
def quarter_improve_early_stop(res:dict[str, float], best_res:dict[str, float]) -> tuple[bool, dict[str, float]]:
    if best_res == {}:
        return True, res
    unimproved_count = 0

    for key, value in res.items():
        if key == 'MRR':
            continue
        if key not in best_res:
            raise ValueError(f"Key {key} not in best_res")
        if value <= best_res[key]:
            unimproved_count += 1
            logger.info(f"{unimproved_count:3d}th metric {key:>10s} did not improve: {value:.5f} <= {best_res[key]:.5f}")
        else:
            logger.info(f"      metric {key:>10s} improved: {value:.5f}")

    if unimproved_count <= len(res) / 4:
        return True, res
    return False, best_res

@register_early_stop('recall_weight')
def recall_weight_early_stop(res:dict[str, float], best_res:dict[str, float]) -> tuple[bool, dict[str, float]]:
    if best_res == {}:
        return True, res
    score = 0.0
    best_score = 0.0
    weights = {
        'ReCall1': 0.4,
        'ReCall5': 0.3,
        'ReCall10': 0.2,
        'ReCall20': 0.1,
    }
    for key, value in res.items():
        if key not in best_res:
            raise ValueError(f"Key {key} not in best_res")
        if key in weights:
            score += value * weights[key]
            best_score += best_res[key] * weights[key]
        if value < best_res[key]:
            logger.info(f"Metric {key:>10s} : {value:.5f} <     {best_res[key]:.5f}")
        elif value > best_res[key]:
            logger.info(f"Metric {key:>10s} : {value:.5f}     > {best_res[key]:.5f}")
        else:
            logger.info(f"Metric {key:>10s} : {value:.5f}   =  {best_res[key]:.5f}")
    if score < best_score:
        logger.info(f"Weighted recall score: {score:.5f} < {best_score:.5f}")
    elif score > best_score:
        logger.info(f"Weighted recall score: {score:.5f} > {best_score:.5f}")
    else:
        logger.info(f"Weighted recall score: {score:.5f} = {best_score:.5f}")
    if score > best_score:
        return True, res
    return False, best_res
