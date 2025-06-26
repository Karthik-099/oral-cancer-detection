import os
import json
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from PIL import Image, ImageTk

class ImageAnnotationTool:
    def __init__(self, root):
        self.root = root
        self.root.title("Surgical Tool Annotation Tool")
        self.root.geometry("1000x700")
        
        # Variables
        self.current_image_path = None
        self.current_image_name = ""
        self.photo_image = None
        self.selected_tool = tk.StringVar(value="0:Cancerous")
        self.coordinates_data = []
        self.json_file_path = None
        self.scale_factor = 1.0
        self.start_x = None
        self.start_y = None
        self.current_rect = None
        
        # Create the layout
        self.create_menu()
        self.create_toolbar()
        self.create_image_area()
        self.create_status_bar()
    
    def create_menu(self):
        menu_bar = tk.Menu(self.root)
        file_menu = tk.Menu(menu_bar, tearoff=0)
        file_menu.add_command(label="Open Image", command=self.open_image_dialog)
        file_menu.add_command(label="Save Annotations", command=self.save_annotations)
        file_menu.add_separator()
        file_menu.add_command(label="Exit", command=self.root.quit)
        menu_bar.add_cascade(label="File", menu=file_menu)
        self.root.config(menu=menu_bar)
    
    def create_toolbar(self):
        toolbar_frame = ttk.Frame(self.root, padding="5")
        toolbar_frame.pack(side=tk.TOP, fill=tk.X)
        
        ttk.Label(toolbar_frame, text="Image Name:").pack(side=tk.LEFT, padx=5)
        self.image_name_entry = ttk.Entry(toolbar_frame, width=30)
        self.image_name_entry.pack(side=tk.LEFT, padx=5)
        ttk.Button(toolbar_frame, text="Open", command=self.open_image_by_name).pack(side=tk.LEFT, padx=5)
        
        ttk.Label(toolbar_frame, text="Tool:").pack(side=tk.LEFT, padx=(20, 5))
        tools = ['0:Cancerous', '1:Non Cancerous']
        tool_dropdown = ttk.Combobox(toolbar_frame, textvariable=self.selected_tool, values=tools, width=15, state="readonly")
        tool_dropdown.pack(side=tk.LEFT, padx=5)
        if tools:
            self.selected_tool.set(tools[0])
        
        ttk.Label(toolbar_frame, text="Scale Factor:").pack(side=tk.LEFT, padx=(20, 5))
        self.scale_entry = ttk.Entry(toolbar_frame, width=10)
        self.scale_entry.insert(0, "2.0")
        self.scale_entry.pack(side=tk.LEFT, padx=5)
        ttk.Button(toolbar_frame, text="Apply Scale", command=self.reload_current_image).pack(side=tk.LEFT, padx=5)
        
        save_button = ttk.Button(toolbar_frame, text="Save Annotations", command=self.save_annotations)
        save_button.pack(side=tk.LEFT, padx=(20, 5))
        
        ttk.Button(toolbar_frame, text="Clear Annotations", command=self.clear_annotations).pack(side=tk.RIGHT, padx=10)
    
    def create_image_area(self):
        self.main_frame = ttk.Frame(self.root)
        self.main_frame.pack(side=tk.TOP, fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        self.canvas_frame = ttk.Frame(self.main_frame)
        self.canvas_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        self.h_scrollbar = ttk.Scrollbar(self.canvas_frame, orient=tk.HORIZONTAL)
        self.h_scrollbar.pack(side=tk.BOTTOM, fill=tk.X)
        
        self.v_scrollbar = ttk.Scrollbar(self.canvas_frame)
        self.v_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        self.canvas = tk.Canvas(
            self.canvas_frame, 
            bd=0, 
            xscrollcommand=self.h_scrollbar.set,
            yscrollcommand=self.v_scrollbar.set,
            bg="lightgray"
        )
        self.canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        self.h_scrollbar.config(command=self.canvas.xview)
        self.v_scrollbar.config(command=self.canvas.yview)
        
        self.canvas.bind("<Button-1>", self.on_canvas_click)
        self.canvas.bind("<B1-Motion>", self.on_canvas_drag)
        self.canvas.bind("<ButtonRelease-1>", self.on_canvas_release)
        
        self.root.bind("<Configure>", self.on_window_resize)
        
        self.display_placeholder()
        
        self.create_annotations_panel()
    
    def create_annotations_panel(self):
        annotations_frame = ttk.LabelFrame(self.main_frame, text="Annotations", padding="5")
        annotations_frame.pack(side=tk.RIGHT, fill=tk.Y, padx=5, pady=5, ipadx=5, ipady=5)
        
        self.tree = ttk.Treeview(
            annotations_frame, 
            columns=("Frame", "Label", "X1", "Y1", "X2", "Y2"), 
            show="headings", 
            height=20
        )
        self.tree.pack(fill=tk.BOTH, expand=True)
        
        self.tree.heading("Frame", text="Frame")
        self.tree.heading("Label", text="Label")
        self.tree.heading("X1", text="X1")
        self.tree.heading("Y1", text="Y1")
        self.tree.heading("X2", text="X2")
        self.tree.heading("Y2", text="Y2")
        
        self.tree.column("Frame", width=100)
        self.tree.column("Label", width=80)
        self.tree.column("X1", width=50)
        self.tree.column("Y1", width=50)
        self.tree.column("X2", width=50)
        self.tree.column("Y2", width=50)
        
        save_button = ttk.Button(annotations_frame, text="Save to JSON", command=self.save_annotations)
        save_button.pack(side=tk.BOTTOM, fill=tk.X, pady=(5, 0))
    
    def create_status_bar(self):
        self.status_var = tk.StringVar()
        self.status_var.set("Ready")
        status_bar = ttk.Label(self.root, textvariable=self.status_var, relief=tk.SUNKEN, anchor=tk.W)
        status_bar.pack(side=tk.BOTTOM, fill=tk.X)
    
    def on_window_resize(self, event):
        if event.widget == self.root and self.current_image_path:
            self.root.after(100, self.reload_current_image)
    
    def reload_current_image(self):
        if self.current_image_path:
            print(f"Reloading image: {self.current_image_path}")  # Debug
            self.load_image(self.current_image_path)
            
    def display_placeholder(self):
        self.canvas.delete("all")
        width = self.canvas.winfo_width()
        height = self.canvas.winfo_height()
        if width < 50:
            width = 400
        if height < 50:
            height = 300
        self.canvas.create_text(
            width // 2, 
            height // 2,
            text="Enter an image name and click 'Open'\nor use File > Open Image to browse",
            fill="gray60",
            font=("Arial", 14)
        )
    
    def open_image_dialog(self):
        file_path = filedialog.askopenfilename(
            title="Select Image",
            filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp *.gif")]
        )
        if file_path:
            self.current_image_path = file_path
            self.current_image_name = os.path.basename(file_path)
            self.image_name_entry.delete(0, tk.END)
            self.image_name_entry.insert(0, self.current_image_name)
            self.set_json_file_path(file_path)
            self.load_annotations()
            self.load_image(file_path)
    
    def open_image_by_name(self):
        image_name = self.image_name_entry.get().strip()
        if not image_name:
            messagebox.showinfo("Info", "Please enter an image name")
            return
        
        # Try direct path
        if os.path.exists(image_name):
            self.current_image_path = image_name
            self.current_image_name = os.path.basename(image_name)
            self.set_json_file_path(image_name)
            self.load_annotations()
            self.load_image(self.current_image_path)
            return
        
        # Try with extensions
        for ext in ['.jpg', '.jpeg', '.png', '.bmp', '.gif']:
            if os.path.exists(image_name + ext):
                self.current_image_path = image_name + ext
                self.current_image_name = os.path.basename(self.current_image_path)
                self.set_json_file_path(self.current_image_path)
                self.load_annotations()
                self.load_image(self.current_image_path)
                return
        
        # Search in data/raw/<patient_id> directories
        base_dir = r"/home/karthik/oral-cancer-detection/data/raw"
        for patient_id in os.listdir(base_dir):
            patient_dir = os.path.join(base_dir, patient_id)
            if os.path.isdir(patient_dir):
                image_path = os.path.join(patient_dir, image_name)
                for ext in ['', '.jpg', '.jpeg', '.png', '.bmp', '.gif']:
                    full_path = image_path + ext
                    if os.path.exists(full_path):
                        self.current_image_path = full_path
                        self.current_image_name = os.path.basename(full_path)
                        self.set_json_file_path(full_path)
                        self.load_annotations()
                        self.load_image(full_path)
                        return
        messagebox.showerror("Error", f"Image '{image_name}' not found in data/raw/<patient_id> directories")
    
    def set_json_file_path(self, image_path):
        directory = os.path.dirname(image_path)
        base_name = os.path.splitext(os.path.basename(image_path))[0]
        self.json_file_path = os.path.join(directory, f"{base_name}_annotations.json")
        print(f"JSON file path set to: {self.json_file_path}")  # Debug
        self.status_var.set(f"JSON will be saved to: {self.json_file_path}")
    
    def load_annotations(self):
        if self.json_file_path and os.path.exists(self.json_file_path):
            try:
                with open(self.json_file_path, 'r') as f:
                    self.coordinates_data = json.load(f)
                for ann in self.coordinates_data:
                    if "image_name" in ann:
                        ann["frame"] = self.extract_frame_number(ann["image_name"])
                        del ann["image_name"]
                    if "tool" in ann:
                        ann["label"] = ann["tool"]
                        del ann["tool"]
                    if "x" in ann and "y" in ann and "x1" not in ann:
                        marker_size = 10
                        ann["x1"] = ann["x"] - marker_size
                        ann["y1"] = ann["y"] - marker_size
                        ann["x2"] = ann["x"] + marker_size
                        ann["y2"] = ann["y"] + marker_size
                        del ann["x"]
                        del ann["y"]
                self.status_var.set(f"Loaded annotations from {self.json_file_path}")
            except json.JSONDecodeError:
                self.coordinates_data = []
                self.status_var.set(f"Could not parse JSON file: {self.json_file_path}")
        else:
            self.coordinates_data = []
    
    def load_image(self, image_path):
        try:
            img = Image.open(image_path)
            print(f"Original image size: {img.width}x{img.height}")  # Debug
            
            try:
                user_scale = float(self.scale_entry.get())
                if user_scale <= 0:
                    raise ValueError("Scale factor must be positive")
            except ValueError:
                user_scale = 2.0
                self.scale_entry.delete(0, tk.END)
                self.scale_entry.insert(0, "2.0")
            print(f"User scale: {user_scale}")  # Debug
            
            # Apply user-specified scale factor directly
            self.scale_factor = user_scale
            new_width = int(img.width * self.scale_factor)
            new_height = int(img.height * self.scale_factor)
            print(f"New image size: {new_width}x{new_height}")  # Debug
            img = img.resize((new_width, new_height), Image.LANCZOS)
            self.photo_image = ImageTk.PhotoImage(img)
            
            self.canvas.delete("all")
            self.canvas.create_image(0, 0, anchor=tk.NW, image=self.photo_image)
            
            self.canvas.config(scrollregion=(0, 0, new_width, new_height))
            
            self.draw_existing_annotations()
            
            self.update_annotations_tree()
            
            self.status_var.set(f"Loaded image: {self.current_image_name} ({new_width}x{new_height}, Original: {img.width}x{img.height}) - Scale: {self.scale_factor:.2f}")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load image: {str(e)}")
            print(f"Error loading image: {str(e)}")  # Debug
    
    def draw_existing_annotations(self):
        if not self.current_image_name:
            return
        frame_number = self.extract_frame_number(self.current_image_name)
        for ann in self.coordinates_data:
            if ann["frame"] == frame_number:
                x1 = ann["x1"] * self.scale_factor
                y1 = ann["y1"] * self.scale_factor
                x2 = ann["x2"] * self.scale_factor
                y2 = ann["y2"] * self.scale_factor
                self.draw_marker(x1, y1, x2, y2, ann["label"])
    
    def extract_frame_number(self, image_path):
        base_name = os.path.splitext(os.path.basename(image_path))[0]
        try:
            frame_number = int(''.join(filter(str.isdigit, base_name)))
            return frame_number
        except ValueError:
            return base_name
    
    def on_canvas_click(self, event):
        if not self.current_image_path:
            messagebox.showinfo("Info", "Please open an image first")
            return
        self.start_x = self.canvas.canvasx(event.x)
        self.start_y = self.canvas.canvasy(event.y)
        self.current_rect = self.canvas.create_rectangle(
            self.start_x, self.start_y, self.start_x, self.start_y,
            outline="red", width=2, dash=(4, 4)
        )
    
    def on_canvas_drag(self, event):
        if not self.current_image_path or self.current_rect is None:
            return
        current_x = self.canvas.canvasx(event.x)
        current_y = self.canvas.canvasy(event.y)
        self.canvas.coords(
            self.current_rect,
            self.start_x, self.start_y, current_x, current_y
        )
        width = abs(current_x - self.start_x) / self.scale_factor
        height = abs(current_y - self.start_y) / self.scale_factor
        self.status_var.set(f"Drawing rectangle: Width={width:.0f}, Height={height:.0f}")
    
    def on_canvas_release(self, event):
        if not self.current_image_path or self.current_rect is None:
            return
        end_x = self.canvas.canvasx(event.x)
        end_y = self.canvas.canvasy(event.y)
        x1 = min(self.start_x, end_x)
        y1 = min(self.start_y, end_y)
        x2 = max(self.start_x, end_x)
        y2 = max(self.start_y, end_y)
        tool = self.selected_tool.get()
        self.canvas.delete(self.current_rect)
        self.draw_marker(x1, y1, x2, y2, tool)
        original_x1 = int(x1 / self.scale_factor)
        original_y1 = int(y1 / self.scale_factor)
        original_x2 = int(x2 / self.scale_factor)
        original_y2 = int(y2 / self.scale_factor)
        frame_number = self.extract_frame_number(self.current_image_path)
        annotation = {
            "frame": frame_number,
            "label": tool,
            "x1": original_x1,
            "y1": original_y1,
            "x2": original_x2,
            "y2": original_y2
        }
        self.coordinates_data.append(annotation)
        self.tree.insert("", tk.END, values=(
            frame_number,
            tool,
            original_x1,
            original_y1,
            original_x2,
            original_y2
        ))
        width = abs(original_x2 - original_x1)
        height = abs(original_y2 - original_y1)
        self.status_var.set(f"Added {tool} rectangle at ({original_x1}, {original_y1}) to ({original_x2}, {original_y2}) - Size: {width}x{height}")
        self.current_rect = None
        self.start_x = None
        self.start_y = None
    
    def draw_marker(self, x1, y1, x2, y2, tool):
        self.canvas.create_rectangle(
            x1, y1, x2, y2,
            fill="",
            outline="red",
            width=2
        )
        self.canvas.create_text(
            x1 + 5, y1 + 15,
            text=tool,
            fill="red",
            font=("Arial", 10, "bold"),
            anchor=tk.NW
        )
    
    def save_annotations(self):
        if not self.json_file_path:
            messagebox.showinfo("Info", "Please open an image first")
            return
        try:
            directory = os.path.dirname(self.json_file_path)
            if directory and not os.path.exists(directory):
                os.makedirs(directory)
            frame_number = self.extract_frame_number(self.current_image_path)
            current_frame_annotations = [
                ann for ann in self.coordinates_data 
                if ann.get("frame") == frame_number
            ]
            with open(self.json_file_path, 'w') as f:
                json.dump(current_frame_annotations, f, indent=4)
            messagebox.showinfo("Success", f"Annotations saved to {self.json_file_path}")
            self.status_var.set(f"Annotations saved to {self.json_file_path}")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to save annotations: {str(e)}")
    
    def update_annotations_tree(self):
        for item in self.tree.get_children():
            self.tree.delete(item)
        frame_number = self.extract_frame_number(self.current_image_name)
        current_image_annotations = [
            ann for ann in self.coordinates_data 
            if ann["frame"] == frame_number
        ]
        for ann in current_image_annotations:
            self.tree.insert("", tk.END, values=(
                ann["frame"],
                ann["label"],
                ann["x1"],
                ann["y1"],
                ann["x2"],
                ann["y2"]
            ))
    
    def clear_annotations(self):
        if not self.current_image_path:
            messagebox.showinfo("Info", "Please open an image first")
            return
        frame_number = self.extract_frame_number(self.current_image_path)
        if messagebox.askyesno("Confirm", f"Clear all annotations for frame {frame_number}?"):
            self.coordinates_data = [
                ann for ann in self.coordinates_data 
                if ann.get("frame") != frame_number
            ]
            for item in self.tree.get_children():
                self.tree.delete(item)
            if self.current_image_path:
                self.load_image(self.current_image_path)
            self.status_var.set(f"Cleared annotations for frame {frame_number}")

if __name__ == "__main__":
    root = tk.Tk()
    app = ImageAnnotationTool(root)
    root.mainloop()