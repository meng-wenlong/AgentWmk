from .task_verification import TaskVerificationWatermark
from .version_check import VersionCheckWatermark
from .environ_detection import EnvironDetectionWatermark
from .file_create_check import FileCreateCheckWatermark
from .check_network import CheckNetworkWatermark
from .visit_web import VisitWebWatermark
from .codemark import CodeMarkWatermark
from .autopoison import AutoPoisonWatermark
from .deadcode import DeadCodeWatermark
from .style_transfer import StyleTransferWatermark

all_watermarks = [
    TaskVerificationWatermark,
    VersionCheckWatermark,
    EnvironDetectionWatermark,
    FileCreateCheckWatermark,
    CheckNetworkWatermark,
    VisitWebWatermark,
    CodeMarkWatermark,
    AutoPoisonWatermark,
    DeadCodeWatermark,
    StyleTransferWatermark,
]