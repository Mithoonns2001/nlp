import sys
import subprocess
import tempfile
import os
from PyQt5.QtCore import *
from PyQt5.QtGui import QKeySequence, QIcon
from PyQt5.QtGui import *
from PyQt5.QtWidgets import (QMainWindow, QAction, QMenu, QMenuBar, QDockWidget, QTreeView,
                             QFileSystemModel, QLineEdit, QTextBrowser, QApplication, QTextEdit, QStatusBar,
                             QInputDialog, QMessageBox, QFileDialog, QToolBar, QVBoxLayout, QWidget, QDialog, QPushButton, QLabel, QComboBox )
from PyQt5.QtWidgets import *
class OutputPanel(QPlainTextEdit):
    def __init__(self, *args, **kwargs):
        super(OutputPanel, self).__init__(*args, **kwargs)
        self.setReadOnly(True)

    def write(self, text):
        self.insertPlainText(text)

    def flush(self):
        pass

class CodeEditor(QMainWindow):
    def __init__(self):
        super().__init__()

        self.init_ui()

    def init_ui(self):
        self.init_menu_bar()
        self.init_toolbar() 
        self.init_code_editor()
        self.init_project_explorer()
        self.init_natural_language_input()
        self.init_output_panel()
        self.init_solution_panel()
        self.init_status_bar()

        self.setWindowTitle("NLP Code Editor")
        self.setGeometry(100, 100, 800, 600)

    def init_menu_bar(self):
        menu_bar = self.menuBar()

        file_menu = menu_bar.addMenu("File")
        edit_menu = menu_bar.addMenu("Edit")
        view_menu = menu_bar.addMenu("View")
        tools_menu = menu_bar.addMenu("Tools")
        help_menu = menu_bar.addMenu("Help")

        new_project_action = QAction("New Project", self)
        new_project_action.triggered.connect(self.new_project)
        file_menu.addAction(new_project_action)

        open_project_action = QAction("Open Project", self)
        open_project_action.triggered.connect(self.open_project)
        file_menu.addAction(open_project_action)

        save_project_action = QAction("Save Project", self)
        save_project_action.triggered.connect(self.save_project)
        file_menu.addAction(save_project_action)

        close_action = QAction("Close", self)
        close_action.setShortcut(QKeySequence.Quit)
        close_action.triggered.connect(self.close)
        file_menu.addAction(close_action)

        undo_action = QAction("Undo", self)
        undo_action.setShortcut(QKeySequence.Undo)
        undo_action.triggered.connect(self.undo)
        edit_menu.addAction(undo_action)

        redo_action = QAction("Redo", self)
        redo_action.setShortcut(QKeySequence.Redo)
        redo_action.triggered.connect(self.redo)
        edit_menu.addAction(redo_action)

        cut_action = QAction("Cut", self)
        cut_action.setShortcut(QKeySequence.Cut)
        cut_action.triggered.connect(self.cut)
        edit_menu.addAction(cut_action)

        save_action = QAction("Save", self)
        save_action.setShortcut("Ctrl+S")
        save_action.triggered.connect(self.save_current_tab)
        file_menu.addAction(save_action)

        copy_action = QAction("Copy", self)
        copy_action.setShortcut(QKeySequence.Copy)
        copy_action.triggered.connect(self.copy)
        edit_menu.addAction(copy_action)

        paste_action = QAction("Paste", self)
        paste_action.setShortcut(QKeySequence.Paste)
        paste_action.triggered.connect(self.paste)
        edit_menu.addAction(paste_action)

        find_replace_action = QAction("Find/Replace", self)
        find_replace_action.setShortcut(QKeySequence.Find)
        find_replace_action.triggered.connect(self.find_replace)
        edit_menu.addAction(find_replace_action)

        toggle_project_explorer_action = QAction("Project Explorer", self, checkable=True)
        toggle_project_explorer_action.setChecked(True)
        toggle_project_explorer_action.triggered.connect(self.toggle_project_explorer)
        view_menu.addAction(toggle_project_explorer_action)

        toggle_output_panel_action = QAction("Output Panel", self, checkable=True)
        toggle_output_panel_action.setChecked(True)
        toggle_output_panel_action.triggered.connect(self.toggle_output_panel)
        view_menu.addAction(toggle_output_panel_action)

        toggle_solution_panel_action = QAction("Solution Panel", self, checkable=True)
        toggle_solution_panel_action.setChecked(True)
        toggle_solution_panel_action.triggered.connect(self.toggle_solution_panel)
        view_menu.addAction(toggle_solution_panel_action)

        library_management_action = QAction("Library Management", self)
        library_management_action.triggered.connect(self.library_management)
        tools_menu.addAction(library_management_action)

        model_deployment_action =model_deployment_action = QAction("Model Deployment", self)
        model_deployment_action.triggered.connect(self.model_deployment)
        tools_menu.addAction(model_deployment_action)

        settings_action = QAction("Settings", self)
        settings_action.triggered.connect(self.settings)
        tools_menu.addAction(settings_action)

        documentation_action = QAction("Documentation", self)
        documentation_action.triggered.connect(self.documentation)
        help_menu.addAction(documentation_action)

        tutorials_action = QAction("Tutorials", self)
        tutorials_action.triggered.connect(self.tutorials)
        help_menu.addAction(tutorials_action)

        support_forum_action = QAction("Support Forum", self)
        support_forum_action.triggered.connect(self.support_forum)
        help_menu.addAction(support_forum_action)

    def init_toolbar(self):
        toolbar = self.addToolBar("Toolbar")
        toolbar.setMovable(False)

        new_project_action = QAction("New Project", self)
        new_project_action.triggered.connect(self.new_project)
        toolbar.addAction(new_project_action)

        open_project_action = QAction("Open Project", self)
        open_project_action.triggered.connect(self.open_project)
        toolbar.addAction(open_project_action)

        save_project_action = QAction("Save Project", self)
        save_project_action.triggered.connect(self.save_project)
        toolbar.addAction(save_project_action)

        run_code_action = QAction("Run Code", self)
        run_code_action.triggered.connect(self.run_code)
        toolbar.addAction(run_code_action)

        settings_action = QAction("Settings", self)
        settings_action.triggered.connect(self.settings)
        toolbar.addAction(settings_action)

        solution_action = QAction(QIcon("solution_icon.png"), "Solution", self)
        solution_action.triggered.connect(self.show_solution)
        toolbar.addAction(solution_action)



    def init_code_editor(self):
        self.tab_widget = QTabWidget()
        self.tab_widget.setTabsClosable(True)  # Add close button to each tab
        self.tab_widget.tabCloseRequested.connect(self.close_tab)  # Connect close button to a function
        self.setCentralWidget(self.tab_widget)

    def on_file_clicked(self, index):
        # Get the file path from the index
        file_path = self.file_system_model.filePath(index)
        
        # Check if the selected item is a file (not a directory)
        if os.path.isfile(file_path):
            with open(file_path, "r") as file:
                file_contents = file.read()
                self.code_editor.setPlainText(file_contents)


    # Modify init_project_explorer to connect doubleClicked signal
    # def init_project_explorer(self):
    #     self.project_explorer = QDockWidget("Project Explorer", self)
    #     self.addDockWidget(Qt.LeftDockWidgetArea, self.project_explorer)

    #     self.file_system_model = QFileSystemModel()
    #     self.file_system_model.setRootPath("")

    #     self.tree_view = QTreeView()
    #     self.tree_view.setModel(self.file_system_model)
    #     self.tree_view.setRootIndex(self.file_system_model.index("."))
    #     self.project_explorer.setWidget(self.tree_view)

    #     # Connect the clicked signal to the custom slot
    #     self.tree_view.clicked.connect(self.on_file_clicked)

    def init_project_explorer(self):
        self.project_explorer = QDockWidget("Project Explorer", self)
        self.addDockWidget(Qt.LeftDockWidgetArea, self.project_explorer)

        file_system_model = QFileSystemModel()
        file_system_model.setRootPath("")

        tree_view = QTreeView()
        tree_view.setModel(file_system_model)
        tree_view.setRootIndex(file_system_model.index("."))
        tree_view.doubleClicked.connect(self.open_file_in_tab)
        tree_view.setContextMenuPolicy(Qt.CustomContextMenu)
        tree_view.customContextMenuRequested.connect(self.show_project_explorer_context_menu)
        self.project_explorer.setWidget(tree_view)

    def show_project_explorer_context_menu(self, point):
        index = self.project_explorer.widget().indexAt(point)
        if not index.isValid():
            return

        menu = QMenu()
        new_file_action = QAction("New File", self)
        new_file_action.triggered.connect(lambda: self.create_new_file(index))
        menu.addAction(new_file_action)

        menu.exec_(self.project_explorer.widget().viewport().mapToGlobal(point))

    def create_new_file(self, index):
        file_path = self.project_explorer.widget().model().filePath(index)
        if os.path.isfile(file_path):
            file_path = os.path.dirname(file_path)

        file_name, ok = QInputDialog.getText(self, "New File", "Enter file name with extension:")
        if ok and file_name:
            new_file_path = os.path.join(file_path, file_name)
            with open(new_file_path, "w"):
                pass
            self.project_explorer.widget().setRootIndex(self.project_explorer.widget().model().index("."))


    def open_file_in_tab(self, index):
        file_path = self.project_explorer.widget().model().filePath(index)
        if not os.path.isfile(file_path):
            return

        for i in range(self.tab_widget.count()):
            if self.tab_widget.widget(i).property("file_path") == file_path:
                self.tab_widget.setCurrentIndex(i)
                break
        else:
            with open(file_path, "r") as file:
                content = file.read()

            new_tab = QPlainTextEdit()
            new_tab.setPlainText(content)
            new_tab.setProperty("file_path", file_path)
            tab_index = self.tab_widget.addTab(new_tab, os.path.basename(file_path))
            self.tab_widget.setCurrentIndex(tab_index)

    def close_tab(self, index):
        self.tab_widget.removeTab(index)
         
    # New slot to handle opening files from the project explorer
    def open_file_from_explorer(self, index):
        file_path = self.sender().model().filePath(index)
        file_info = QFileInfo(file_path)

        if file_info.isFile():
            with open(file_path, "r") as file:
                file_content = file.read()

            self.code_editor.setPlainText(file_content)
            self.setWindowTitle(f"NLP Code Editor - {file_path}")


    # def init_natural_language_input(self):
    #     # self.natural_language_input = QLineEdit(self)
    #     # self.natural_language_input.setPlaceholderText("Enter natural language commands here...")
    #     # self.natural_language_input.setToolTip("Enter natural language commands here and press Enter to generate code.")
    #     # self.natural_language_input.returnPressed.connect(self.generate_code_from_natural_language)

    #     # natural_language_input_toolbar = QToolBar("Natural Language Input")
    #     # natural_language_input_toolbar.addWidget(self.natural_language_input)
    #     # self.addToolBar(Qt.BottomToolBarArea, natural_language_input_toolbar)
    #     pass
    def init_natural_language_input(self):
            self.natural_language_input = QLineEdit(self)
            self.natural_language_input.setPlaceholderText("Type natural language commands here...")
            self.natural_language_input.returnPressed.connect(self.generate_code)

            natural_language_input_dock = QDockWidget("Natural Language Input", self)
            natural_language_input_dock.setWidget(self.natural_language_input)
            self.addDockWidget(Qt.TopDockWidgetArea, natural_language_input_dock)
            
    # def init_output_panel(self):
    #     self.output_panel = QDockWidget("Output Panel", self)
    #     self.addDockWidget(Qt.BottomDockWidgetArea, self.output_panel)

    #     output_browser = QTextBrowser()
    #     self.output_panel.setWidget(output_browser)

    # def init_output_panel(self):
    #     self.output_panel = OutputPanel()
    #     output_dock = QDockWidget("Output", self)
    #     output_dock.setWidget(self.output_panel)
    #     self.addDockWidget(Qt.BottomDockWidgetArea, output_dock)
    def init_output_panel(self):
        self.output_panel = QDockWidget("Output", self)
        self.addDockWidget(Qt.BottomDockWidgetArea, self.output_panel)

        self.output_text_edit = QTextEdit()
        self.output_text_edit.setReadOnly(True)
        self.output_text_edit.setStyleSheet("QTextEdit { background-color: black; color: white; }")
        self.output_panel.setWidget(self.output_text_edit)

    def init_solution_panel(self):
        self.solution_panel = QDockWidget("Solution", self)
        self.addDockWidget(Qt.BottomDockWidgetArea, self.solution_panel)

        solution_widget = QWidget()
        solution_layout = QVBoxLayout()

        self.solution_text_edit = QTextEdit()
        self.solution_text_edit.setReadOnly(True)
        self.solution_text_edit.setStyleSheet("QTextEdit { background-color: green; color: black; }")
        solution_layout.addWidget(self.solution_text_edit)

        self.solution_button = QPushButton("Get Solution")
        self.solution_button.clicked.connect(self.show_solution)
        solution_layout.addWidget(self.solution_button)

        solution_widget.setLayout(solution_layout)
        self.solution_panel.setWidget(solution_widget)

            
    def show_solution(self):
        # Get the error text from the output panel
        error_text = self.output_text_edit.toPlainText()

        # Use your trained model to get the solution for the error
        solution = self.get_solution_from_model(error_text)

        # Display the solution in the Solution panel
        self.solution_text_edit.setPlainText(solution)

    def get_solution_from_model(self, error_text):
        # Replace this placeholder code with your actual model usage
        solution = "Solution for the error: " + error_text
        return solution


    def init_status_bar(self):
        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)

    def new_project(self):
        project_name, ok = QInputDialog.getText(self, "New Project", "Project Name:")

        if ok and project_name:
            # Create a new project directory and set it as the current project path
            self.project_path = f"./{project_name}"
            os.makedirs(self.project_path, exist_ok=True)

            # Update the project explorer to show the new project directory
            file_system_model = QFileSystemModel()
            file_system_model.setRootPath(self.project_path)

            tree_view = QTreeView()
            tree_view.setModel(file_system_model)
            tree_view.setRootIndex(file_system_model.index(self.project_path))
            self.project_explorer.setWidget(tree_view)

    def open_project(self):
        project_path = QFileDialog.getExistingDirectory(self, "Open Project")

        if project_path:
            # Set the selected directory as the current project path
            self.project_path = project_path

            # Update the project explorer to show the opened project directory
            file_system_model = QFileSystemModel()
            file_system_model.setRootPath(self.project_path)

            tree_view = QTreeView()
            tree_view.setModel(file_system_model)
            tree_view.setRootIndex(file_system_model.index(self.project_path))
            self.project_explorer.setWidget(tree_view)

    def save_project(self):
        if self.project_path:
            file_name, _ = QFileDialog.getSaveFileName(self, "Save Project", self.project_path)

            if file_name:
                with open(file_name, "w") as file:
                    file.write(self.code_editor.toPlainText())
        else:
            QMessageBox.warning(self, "Warning", "No project is open. Please open or create a project first.")

    def save_current_tab(self):
        current_tab = self.tab_widget.currentWidget()
        if current_tab is None:
            return

        file_path = current_tab.property("file_path")
        if not file_path:
            return

        code = current_tab.toPlainText()
        with open(file_path, "w") as file:
            file.write(code)


    # def run_code(self):
    #     if not self.project_path:
    #         QMessageBox.warning(self, "Warning", "No project is open. Please open or create a project first.")
    #         return

    #     # Save the current contents of the code editor to a temporary file
    #     temp_file = tempfile.NamedTemporaryFile(delete=False)
    #     with open(temp_file.name, 'w') as file:
    #         file.write(self.code_editor.toPlainText())

    #     # Run the code using the appropriate interpreter (assuming Python for this example)
    #     process = subprocess.Popen(["python", temp_file.name], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    #     stdout, stderr = process.communicate()

    #     # Display the output and/or error messages in the output panel
    #     self.output_panel.clear()
    #     if stdout:
    #         self.output_panel.appendPlainText(stdout.decode('utf-8'))
    #     if stderr:
    #         self.output_panel.appendPlainText(stderr.decode('utf-8'))

    #     # Clean up the temporary file
    #     os.remove(temp_file.name)
############
    # def run_code(self):
    #     code = self.code_editor.toPlainText()

    #     # Redirect stdout to the output panel
    #     sys.stdout = self.output_panel

    #     # Clear the output panel before running the code
    #     self.output_panel.clear()

    #     # Execute the code
    #     try:
    #         exec(code)
    #     except Exception as e:
    #         print(f"Error: {e}")

    #     # Restore the original stdout
    #     sys.stdout = sys.__stdout__


    def run_code(self):
        current_tab = self.tab_widget.currentWidget()
        if current_tab is None:
            return

        code = current_tab.toPlainText()
        if not code:
            return

        # Save the code to a temporary file
        with open("temp.py", "w") as temp_file:
            temp_file.write(code)

        # Create a custom input function that reads from the QLineEdit
        def custom_input(prompt):
            self.output_text_edit.append(prompt)
            input_line_edit = QLineEdit(self.output_panel)
            input_line_edit.returnPressed.connect(lambda: input_line_edit.setProperty("done", True))
            self.output_panel.setWidget(input_line_edit)
            input_line_edit.setFocus()

            while not input_line_edit.property("done"):
                QApplication.processEvents()

            value = input_line_edit.text()
            self.output_panel.setWidget(self.output_text_edit)
            return value

        # Redirect the input function to our custom input function
        sys.stdin.readline = custom_input

        # Run the code in a separate process and capture the output
        process = subprocess.Popen(["python", "temp.py"], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        stdout, stderr = process.communicate()

        # Display the output in the output panel
        self.output_text_edit.clear()
        self.output_text_edit.append("Output:\n" + stdout)
        if stderr:
            self.output_text_edit.append("Error:\n" + stderr)

        # Restore the original input function
        sys.stdin.readline = sys.__stdin__.readline


    def settings(self):
        settings_dialog = QDialog(self)
        settings_dialog.setWindowTitle("Settings")

        # Add widgets and layouts for the settings dialog
        # For example, you can add options for changing the theme or font of the code editor
        layout = QVBoxLayout()

        theme_label = QLabel("Theme:")
        theme_combobox = QComboBox()
        theme_combobox.addItems(["Default", "Dark"])
        layout.addWidget(theme_label)
        layout.addWidget(theme_combobox)

        font_label = QLabel("Font:")
        font_combobox = QFontComboBox()
        layout.addWidget(font_label)
        layout.addWidget(font_combobox)

        ok_button = QPushButton("OK")
        ok_button.clicked.connect(settings_dialog.accept)
        layout.addWidget(ok_button)

        settings_dialog.setLayout(layout)
        result = settings_dialog.exec_()

        if result == QDialog.Accepted:
            # Apply the selected settings
            selected_theme = theme_combobox.currentText()
            selected_font = font_combobox.currentFont()

            if selected_theme == "Default":
                self.code_editor.setStyleSheet("QPlainTextEdit { background-color: white; color: black; }")
            elif selected_theme == "Dark":
                self.code_editor.setStyleSheet("QPlainTextEdit { background-color: #272822; color: #f8f8f2; }")

            self.code_editor.setFont(selected_font)

    def undo(self):
        self.code_editor.undo()

    def redo(self):
        self.code_editor.redo()

    def cut(self):
        self.code_editor.cut()

    def copy(self):
        self.code_editor.copy()

    def paste(self):
        self.code_editor.paste()

    def find_replace(self):
        find_replace_dialog = QDialog(self)
        find_replace_dialog.setWindowTitle("Find & Replace")

        layout = QVBoxLayout()

        find_label = QLabel("Find:")
        find_line_edit = QLineEdit()
        layout.addWidget(find_label)
        layout.addWidget(find_line_edit)

        replace_label = QLabel("Replace with:")
        replace_line_edit = QLineEdit()
        layout.addWidget(replace_label)
        layout.addWidget(replace_line_edit)

        find_button = QPushButton("Find")
        find_button.clicked.connect(lambda: self.find(find_line_edit.text()))
        layout.addWidget(find_button)

        replace_button = QPushButton("Replace")
        replace_button.clicked.connect(lambda: self.replace(find_line_edit.text(), replace_line_edit.text()))
        layout.addWidget(replace_button)

        find_replace_dialog.setLayout(layout)
        find_replace_dialog.exec_()

    def find(self, text):
        if not self.code_editor.find(text):
            QMessageBox.information(self, "Find", "The specified text could not be found.")

    def replace(self, find_text, replace_text):
        cursor = self.code_editor.textCursor()

        if cursor.hasSelection() and cursor.selectedText() == find_text:
            cursor.insertText(replace_text)

        if not self.code_editor.find(find_text):
            QMessageBox.information(self, "Replace", "The specified text could not be found.")


    def toggle_project_explorer(self, checked):
        if checked:
            self.project_explorer.show()
        else:
            self.project_explorer.hide()

    def toggle_output_panel(self, checked):
        if checked:
            self.output_panel.show()
        else:
            self.output_panel.hide()

    def toggle_solution_panel(self, checked):
        if checked:
            self.solution_panel.show()
        else:
            self.solution_panel.hide()

    def library_management(self):
        pass

    def model_deployment(self):
        pass

    def documentation(self):
        pass

    def tutorials(self):
        pass

    def support_forum(self):
        pass

    def generate_code(self):
        pass


if __name__ == "__main__":
    app = QApplication(sys.argv)
    editor = CodeEditor()
    editor.show()

    def close_application():
        response = QMessageBox.question(editor, "Exit", "Are you sure you want to exit?",
                                        QMessageBox.Yes | QMessageBox.No)
        if response == QMessageBox.Yes:
            sys.exit()

    app.aboutToQuit.connect(close_application)
    sys.exit(app.exec_())
   

