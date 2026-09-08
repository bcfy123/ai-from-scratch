import json
import regex as re
from functools import lru_cache

@lru_cache()
def bytes_to_unicode():
    """
    把 0~255 的 byte 映射成可打印的 Unicode 字符
    从而让 BPE 可以直接在字符串上工作
    """
    bs = (
        list(range(ord("!"), ord("~") + 1))
        + list(range(ord("¡"), ord("¬") + 1))
        + list(range(ord("®"), ord("ÿ") + 1))
    )
    cs = bs[:]
    n = 0
    # 不安全的unicode 用别的字符进行映射
    for b in range(2**8):
        if b not in bs:
            bs.append(b)
            cs.append(2**8+n)
            n+=1
    # 放对应的字符
    cs = [chr(n) for n in cs]
    return dict(zip(bs, cs))