'''pytest 共享夹具：自动清掉测试创建的临时目录。

测试里大量 `tempfile.mkdtemp()`，其中不少没挂 `addCleanup` —— 挂上这个
autouse 夹具后，**每一个** `mkdtemp` 都会在测试结束时自动 rmtree，
存量测试一行都不用改。

注意：`python tests/test_promptkit.py`（unittest 直跑）不走 conftest，
那条路的清理靠测试自己的 `addCleanup` / 手动删除。
'''
import gc
import os
import shutil
import stat
import sys
import tempfile
import time

import pytest


def _force_writable(func, path, _exc):
    '''删不动多半是 Windows 的只读属性：git 对象文件、数据库文件。'''
    os.chmod(path, stat.S_IWRITE)
    func(path)


# 3.12 起 `onerror` 改名 `onexc`，两个签名都兼容（第三个参数用不上）
_RMTREE_KWARGS = (
    {"onexc": _force_writable}
    if sys.version_info >= (3, 12)
    else {"onerror": _force_writable}
)


def _remove(path):
    '''删掉一个临时目录，删不掉就去只读 + gc 各重试几次。

    `shutil.rmtree(..., ignore_errors=True)` 会**跳过**错误处理回调，
    所以这里不用 ignore_errors，而是自己吞 OSError。
    '''
    for attempt in range(3):
        if not os.path.exists(path):
            return
        try:
            shutil.rmtree(path, **_RMTREE_KWARGS)
            return
        except OSError:
            if attempt == 2:
                # 清理失败不该把测试搞红，最后兜一次就算了
                shutil.rmtree(path, ignore_errors=True)
                return
            # 测试忘了 close 的 sqlite 连接还占着文件，gc 之后再试
            gc.collect()
            time.sleep(0.01)


@pytest.fixture(autouse=True)
def _auto_cleanup_mkdtemp():
    '''把 `tempfile.mkdtemp` 包一层：这次测试建出来的目录结束时全删。'''
    original = tempfile.mkdtemp
    created = []

    def tracked(*args, **kwargs):
        path = original(*args, **kwargs)
        created.append(path)
        return path

    tempfile.mkdtemp = tracked
    yield
    tempfile.mkdtemp = original
    for path in created:
        _remove(path)
