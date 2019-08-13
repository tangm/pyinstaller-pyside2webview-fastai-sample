import sys
import os
from pathlib import Path
from io import BytesIO
from PySide2.QtWidgets import QApplication
from PySide2.QtWebEngineWidgets import QWebEngineView, QWebEnginePage
from PySide2.QtWebChannel import QWebChannel
from PySide2.QtCore import QUrl, Slot, QObject, QUrl
def script_method(fn, _rcb=None):
    return fn
def script(obj, optimize=True, _frames_up=0, _rcb=None):
    return obj    
import torch.jit
torch.jit.script_method = script_method 
torch.jit.script = script
from fastai.vision import load_learner, open_image

# Some hackery required for pyInstaller
if getattr(sys, 'frozen', False) and sys.platform == 'darwin':
    os.environ['QTWEBENGINEPROCESS_PATH'] = os.path.normpath(os.path.join(
        sys._MEIPASS, 'PySide2', 'Qt', 'lib',
        'QtWebEngineCore.framework', 'Helpers', 'QtWebEngineProcess.app',
        'Contents', 'MacOS', 'QtWebEngineProcess'
    ))

data_dir = Path(os.path.abspath(os.path.dirname(__file__))) / 'data'
model_dir = data_dir / 'models'
export_file_name = 'export.pkl'

def setup_learner():
    try:
        learn = load_learner(model_dir, export_file_name)
        return learn
    except RuntimeError as e:
        if len(e.args) > 0 and 'CPU-only machine' in e.args[0]:
            print(e)
            message = "\n\nThis model was trained with an old version of fastai and will not work in a CPU environment.\n\nPlease update the fastai library in your training environment and export your model again.\n\nSee instructions for 'Returning to work' at https://course.fast.ai."
            raise RuntimeError(message)
        else:
            raise

class Handler(QObject):
    def __init__(self, learn, *args, **kwargs):
        super(Handler, self).__init__(*args, **kwargs)
        self.learn = learn

    @Slot(str, result=str)
    def classify(self, something):
        img_bytes = bytes(map(ord, something))

        img = open_image(BytesIO(img_bytes))
        prediction = self.learn.predict(img)[0]
        return str(prediction)

class WebEnginePage(QWebEnginePage):
    def __init__(self, *args, **kwargs):
        super(WebEnginePage, self).__init__(*args, **kwargs)

    def javaScriptConsoleMessage(self, level, message, lineNumber, sourceId):
        print("WebEnginePage Console: ", level, message, lineNumber, sourceId)

if __name__ == "__main__":

    # Initialize fast.ai learner
    learn = setup_learner()

	# Set up the main application
    app = QApplication([])
    app.setApplicationDisplayName("Bear Classifier")

    # Use a webengine view
    view = QWebEngineView()
    view.resize(1024, 768)

    # Set up backend communication via web channel
    handler = Handler(learn)
    channel = QWebChannel()
    # Make the handler object available, naming it "backend"
    channel.registerObject("backend", handler)

    # Use a custom page that prints console messages to make debugging easier
    page = WebEnginePage()
    page.setWebChannel(channel)
    view.setPage(page)

    # Finally, load our file in the view
    url = QUrl.fromLocalFile(f"{data_dir}/index.html")
    view.load(url)
    view.show()

    app.exec_()
