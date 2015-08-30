import sys
from PyQt4 import QtCore, QtGui
from PyQt4.QtGui import QFileDialog, QScrollArea
import shutil
from test_ui import Ui_MainWindow
import os.path
from sklearn import tree
from sklearn.feature_extraction import DictVectorizer
from sklearn.externals.six import StringIO
import pydot
from collections import defaultdict, OrderedDict
import matplotlib.pyplot as plot


'''
This is the driver for the machine learning GUI. test_ui.py is generated
by the PyQt GUI designer.
'''


#Here are some global variables. This is generally bad design if it can be avoided
#It was done this way in the interest of time
path = ''
alteredPath = ''
ui = None
outputGraph = ''

'''
Called when the user selects a file. Will plot the age distribution
of the adult data set with matplotlib.
'''
def selectFile():
    global path
    path = QFileDialog.getOpenFileName()
    ui.fileUploadText.setText(path + ' uploaded.')

    file = open(path)

    counts = defaultdict(int)
    for line in file:

        list = line.split(',')
        counts[list[0]] += 1

    counts = OrderedDict(sorted(counts.items()))

    #Create the plot
    plot.bar(range(len(counts)), counts.values(), align='center')
    plot.xticks(range(len(counts)), counts.keys())

    fig = plot.gcf()
    fig.set_size_inches(24, 8)

    fig.suptitle('Age distribution', fontsize=18)
    plot.xlabel('Age', fontsize=18)
    plot.ylabel('Number of records', fontsize=18)

    fig.savefig(str(path) + 'plot.png', dpi=60)
    outputGraph = str(path) + 'plot.png'

    pixMap = QtGui.QPixmap(outputGraph)
    ui.inputDataLabel.setPixmap(pixMap)
    
    scrollArea = QScrollArea()
    scrollArea.setWidgetResizable(True)
    scrollArea.setWidget(ui.inputDataLabel)
    scrollArea.setFixedHeight(500)
    scrollArea.horizontalScrollBar().setValue(scrollArea.horizontalScrollBar().value() + 100); 
    hLayout = QtGui.QHBoxLayout()
    hLayout.addWidget(scrollArea)
    ui.uploadTab.setLayout(hLayout)


'''
Defines the movement between tabs
'''
def moveTabs():
    ui.tabWidget.setCurrentIndex((ui.tabWidget.currentIndex() + 1) % 4)

'''
Filters the data set based on the user input
'''
def filterData():
    global path
    if (path):

        #shutil.copyfile(path, path + '_altered')
        alteredFileName = path + '_altered'

        value = str(ui.lineEdit.displayText())

        file = open(path)
        output = open(alteredFileName, 'w+')

        purged = 0

        for line in file:
            if (value not in line):
                output.write(line)
            else:
                purged = purged + 1
        ui.filterText.setText('Removed ' + str(purged) + ' records from data set')

        global alteredPath
        alteredPath = alteredFileName
    else:
        print('Need to upload file')

'''
Runs the decision tree algorithm on the input file
'''
def runDecisionTree():

    #Todo: try not to use global variables
    global path
    global alteredPath

    fileName = ''

    if (alteredPath):
        fileName = alteredPath
    elif (path):
        fileName = path
    else:
        print('Must upload file first')
        return

    x = []
    y = []

    file = open(fileName)
    for line in file:
        line = line.rstrip()
        features = []
        classification = []

        list = line.split(',')
        features = list[0:-1]
        if (features and features[0].strip()):
            x.append(features)

        classification = [list[-1]]
        if (classification and classification[0].strip()):
            y.append(classification)


    ui.progressBar.setValue(25)

    samples = [dict(enumerate(sample)) for sample in x]

    # turn list of dicts into a numpy array
    vect = DictVectorizer(sparse=False)
    x = vect.fit_transform(samples)
    ui.progressBar.setValue(50)

    clf = tree.DecisionTreeClassifier()
    clf = clf.fit(x, y)

    
    with open(fileName + '.dot', 'w') as f:
        f = tree.export_graphviz(clf, out_file=f)
    graph = pydot.graph_from_dot_file(fileName + '.dot')
    graph.write_png(fileName + '.png')
    global outputGraph
    outputGraph = fileName + '.png'

    ui.progressBar.setValue(75)
    

    pixMap = QtGui.QPixmap(outputGraph)
    #ui.outputLabel.setPixmap(pixMap.scaled(ui.outputTab.size(), QtCore.Qt.KeepAspectRatio))
    ui.outputLabel.setPixmap(pixMap)
    #ui.outputLabel.setScaledContents(True)

    
    scrollArea = QScrollArea()
    scrollArea.setWidgetResizable(True)
    scrollArea.setWidget(ui.outputLabel)
    scrollArea.setFixedHeight(525)
    scrollArea.horizontalScrollBar().setValue(scrollArea.horizontalScrollBar().value() + 3400); 
    hLayout = QtGui.QHBoxLayout()
    hLayout.addWidget(scrollArea)
    ui.outputTab.setLayout(hLayout)

    ui.progressBar.setValue(100)

    ui.algorithmText.setText('Built decision tree')

'''
Wrapper for the PyQt GUI file
'''
class MyForm(QtGui.QMainWindow):
    def __init__(self, parent=None):
        QtGui.QWidget.__init__(self, parent)
        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)
        self.ui.uploadButton.clicked.connect(selectFile)
        self.ui.tabWidget.setCurrentIndex(0)
        global ui
        ui = self.ui
        self.ui.nextButton1.clicked.connect(moveTabs)
        self.ui.nextButton2.clicked.connect(moveTabs)
        self.ui.nextButton3.clicked.connect(moveTabs)
        self.ui.homeButton.clicked.connect(moveTabs)
        self.ui.filterButton.clicked.connect(filterData)
        self.ui.runAlgorithmButton.clicked.connect(runDecisionTree)
        self.ui.progressBar.setValue(0)
        self.setWindowTitle('Pykit-Learn')

'''
Here is code run at start time
'''
if __name__ == "__main__":
    app = QtGui.QApplication(sys.argv)
    myapp = MyForm()
    myapp.show()
    sys.exit(app.exec_())

