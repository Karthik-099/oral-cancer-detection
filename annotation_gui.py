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
        self.selected_tool = tk.StringVar(value="0:Cancerous")  # Default to first surgical tool
        self.coordinates_data = []
        self.json_file_path = None  # Will be set dynamically based on image path
        self.scale_factor = 1.0  # Add scale factor tracking
        self.start_x = None  # For drag-to-draw
        self.start_y = None  # For drag-to-draw
        self.current_rect = None  # Track the current rectangle being drawn
        
        # Create the layout
        self.create_menu()
        self.create_toolbar()
        self.create_image_area()
        self.create_status_bar()
    
    def create_menu(self):
        # Main menu
        menu_bar = tk.Menu(self.root)
        
        # File menu
        file_menu = tk.Menu(menu_bar, tearoff=0)
        file_menu.add_command(label="Open Image", command=self.open_image_dialog)
        file_menu.add_command(label="Save Annotations", command=self.save_annotations)
        file_menu.add_separator()
        file_menu.add_command(label="Exit", command=self.root.quit)
        
        menu_bar.add_cascade(label="File", menu=file_menu)
        self.root.config(menu=menu_bar)
    
    def create_toolbar(self):
        # Frame for image name entry and tool selection
        toolbar_frame = ttk.Frame(self.root, padding="5")
        toolbar_frame.pack(side=tk.TOP, fill=tk.X)
        
        # Image name entry
        ttk.Label(toolbar_frame, text="Image Name:").pack(side=tk.LEFT, padx=5)
        self.image_name_entry = ttk.Entry(toolbar_frame, width=30)
        self.image_name_entry.pack(side=tk.LEFT, padx=5)
        ttk.Button(toolbar_frame, text="Open", command=self.open_image_by_name).pack(side=tk.LEFT, padx=5)
        
        # Tool selection
        ttk.Label(toolbar_frame, text="Tool:").pack(side=tk.LEFT, padx=(20, 5))
        tools = ['0:Cancerous', '1:Non Cancerous']
        tool_dropdown = ttk.Combobox(toolbar_frame, textvariable=self.selected_tool, values=tools, width=15, state="readonly")
        tool_dropdown.pack(side=tk.LEFT, padx=5)
        if tools:
            self.selected_tool.set(tools[0])
        
        # Scale factor entry
        ttk.Label(toolbar_frame, text="Scale Factor:").pack(side=tk.LEFT, padx=(20, 5))
        self.scale_entry = ttk.Entry(toolbar_frame, width=10)
        self.scale_entry.insert(0, "2.0")  # Default scale factor
        self.scale_entry.pack(side=tk.LEFT, padx=5)
        ttk.Button(toolbar_frame, text="Apply Scale", command=self.reload_current_image).pack(side=tk.LEFT, padx=5)
        
        # Save button
        save_button = ttk.Button(toolbar_frame, text="Save Annotations", command=self.save_annotations)
        save_button.pack(side=tk.LEFT, padx=(20, 5))
        
        # Clear button
        ttk.Button(toolbar_frame, text="Clear Annotations", command=self.clear_annotations).pack(side=tk.RIGHT, padx=10)
    
    def create_image_area(self):
        # Frame to hold image and scrollbars
        self.main_frame = ttk.Frame(self.root)
        self.main_frame.pack(side=tk.TOP, fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Canvas for displaying the image with scrollbars
        self.canvas_frame = ttk.Frame(self.main_frame)
        self.canvas_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        # Scrollbars
        self.h_scrollbar = ttk.Scrollbar(self.canvas_frame, orient=tk.HORIZONTAL)
        self.h_scrollbar.pack(side=tk.BOTTOM, fill=tk.X)
        
        self.v_scrollbar = ttk.Scrollbar(self.canvas_frame)
        self.v_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        # Canvas
        self.canvas = tk.Canvas(
            self.canvas_frame, 
            bd=0, 
            xscrollcommand=self.h_scrollbar.set,
            yscrollcommand=self.v_scrollbar.set,
            bg="lightgray"
        )
        self.canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        # Configure scrollbars
        self.h_scrollbar.config(command=self.canvas.xview)
        self.v_scrollbar.config(command=self.canvas.yview)
        
        # Bind mouse events for dragging
        self.canvas.bind("<Button-1>", self.on_canvas_click)
        self.canvas.bind("<B1-Motion>", self.on_canvas_drag)
        self.canvas.bind("<ButtonRelease-1>", self.on_canvas_release)
        
        # Bind resize event to update image
        self.root.bind("<Configure>", self.on_window_resize)
        
        # Display placeholder
        self.display_placeholder()
        
        # Create annotations panel
        self.create_annotations_panel()
    
    def create_annotations_panel(self):
        # Annotations panel
        annotations_frame = ttk.LabelFrame(self.main_frame, text="Annotations", padding="5")
        annotations_frame.pack(side=tk.RIGHT, fill=tk.Y, padx=5, pady=5, ipadx=5, ipady=5)
        
        # Create treeview for annotations
        self.tree = ttk.Treeview(
            annotations_frame, 
            columns=("Frame", "Label", "X1", "Y1", "X2", "Y2"), 
            show="headings", 
            height=20
        )
        self.tree.pack(fill=tk.BOTH, expand=True)
        
        # Set column headings
        self.tree.heading("Frame", text="Frame")
        self.tree.heading("Label", text="Label")
        self.tree.heading("X1", text="X1")
        self.tree.heading("Y1", text="Y1")
        self.tree.heading("X2", text="X2")
        self.tree.heading("Y2", text="Y2")
        
        # Set column widths
        self.tree.column("Frame", width=100)
        self.tree.column("Label", width=80)
        self.tree.column("X1", width=50)
        self.tree.column("Y1", width=50)
        self.tree.column("X2", width=50)
        self.tree.column("Y2", width=50)
        
        # Add a save button in the annotations panel
        save_button = ttk.Button(annotations_frame, text="Save to JSON", command=self.save_annotations)
        save_button.pack(side=tk.BOTTOM, fill=tk.X, pady=(5, 0))
    
    def create_status_bar(self):
        # Status bar at the bottom
        self.status_var = tk.StringVar()
        self.status_var.set("Ready")
        status_bar = ttk.Label(self.root, textvariable=self.status_var, relief=tk.SUNKEN, anchor=tk.W)
        status_bar.pack(side=tk.BOTTOM, fill=tk.X)
    
    def on_window_resize(self, event):
        # Only handle resize events for the main window
        if event.widget == self.root and self.current_image_path:
            # Add small delay to prevent multiple redraws
            self.root.after(100, self.reload_current_image)
    
    def reload_current_image(self):
        if self.current_image_path:
            self.load_image(self.current_image_path)
            
    def display_placeholder(self):
        # Clear canvas and display a message
        self.canvas.delete("all")
        
        # Get canvas dimensions
        width = self.canvas.winfo_width()
        height = self.canvas.winfo_height()
        
        # Ensure we have some space even on initial load
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
        # Open file dialog to select an image
        file_path = filedialog.askopenfilename(
            title="Select Image",
            filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp *.gif")]
        )
        
        if file_path:
            self.current_image_path = file_path
            self.current_image_name = os.path.basename(file_path)
            self.image_name_entry.delete(0, tk.END)
            self.image_name_entry.insert(0, self.current_image_name)
            
            # Set JSON file path based on the image path
            self.set_json_file_path(file_path)
            
            # Load annotations if the JSON file exists
            self.load_annotations()
            
            # Load the image
            self.load_image(file_path)
    
    def open_image_by_name(self):
        # Get the image name from the entry
        image_name = self.image_name_entry.get().strip()
        if not image_name:
            messagebox.showinfo("Info", "Please enter an image name")
            return
        
        # Look for the image in the current directory and subdirectories
        found = False
        
        # First try direct path if it includes directories
        if os.path.exists(image_name):
            self.current_image_path = image_name
            self.current_image_name = os.path.basename(image_name)
            self.set_json_file_path(image_name)
            self.load_annotations()
            self.load_image(self.current_image_path)
            return
            
        # Try with different extensions
        for ext in ['.jpg', '.jpeg', '.png', '.bmp', '.gif']:
            if os.path.exists(image_name + ext):
                self.current_image_path = image_name + ext
                self.current_image_name = os.path.basename(self.current_image_path)
                self.set_json_file_path(self.current_image_path)
                self.load_annotations()
                self.load_image(self.current_image_path)
                return
        
        # If still not found, search directories recursively (up to a reasonable depth)
        if not found:
            messagebox.showinfo("Error", f"Image '{image_name}' not found in the current directory")
    
    def set_json_file_path(self, image_path):
        """Set the JSON file path based on the image path"""
        # Get the directory and base name without extension
        directory = os.path.dirname(image_path)
        base_name = os.path.splitext(os.path.basename(image_path))[0]
        
        # Create the JSON path in the same directory with the same base name
        self.json_file_path = os.path.join('Annotations', directory, f"{base_name}.json")
        
        # Update status bar
        self.status_var.set(f"JSON will be saved to: {self.json_file_path}")
    
    def load_annotations(self):
        """Load annotations from the JSON file if it exists"""
        if self.json_file_path and os.path.exists(self.json_file_path):
            try:
                with open(self.json_file_path, 'r') as f:
                    self.coordinates_data = json.load(f)
                    
                # Convert any existing point-based annotations to rectangle-based (if needed)
                for ann in self.coordinates_data:
                    if "image_name" in ann:
                        ann["frame"] = self.extract_frame_number(ann["image_name"])
                        del ann["image_name"]
                    if "tool" in ann:
                        ann["label"] = ann["tool"]
                        del ann["tool"]
                    if "x" in ann and "y" in ann and "x1" not in ann:
                        # Convert point to small rectangle (e.g., 20x20 pixels centered on x,y)
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
            # Load the image using PIL
            img = Image.open(image_path)
            
            # Get canvas dimensions
            canvas_width = self.canvas_frame.winfo_width()
            canvas_height = self.canvas_frame.winfo_height()
            
            # Ensure canvas has proper size (needed on first load)
            if canvas_width < 50 or canvas_height < 50:
                self.root.update_idletasks()  # Force geometry update
                canvas_width = self.canvas_frame.winfo_width()
                canvas_height = self.canvas_frame.winfo_height()
            
            # Get user-specified scale factor
            try:
                user_scale = float(self.scale_entry.get())
                if user_scale <= 0:
                    raise ValueError("Scale factor must be positive")
            except ValueError:
                user_scale = 2.0  # Fallback to default if invalid
                self.scale_entry.delete(0, tk.END)
                self.scale_entry.insert(0, "2.0")
            
            # Calculate scaling factor to fit or upscale the image
            width_ratio = canvas_width / img.width
            height_ratio = canvas_height / img.height
            scale_factor = min(width_ratio, height_ratio, user_scale)  # Use user-specified scale
            
            # Scale the image
            new_width = int(img.width * scale_factor)
            new_height = int(img.height * scale_factor)
            img = img.resize((new_width, new_height), Image.LANCZOS)
            self.scale_factor = scale_factor
            
            self.photo_image = ImageTk.PhotoImage(img)
            
            # Clear canvas and display the image
            self.canvas.delete("all")
            self.canvas.create_image(0, 0, anchor=tk.NW, image=self.photo_image)
            
            # Configure canvas scrolling region
            self.canvas.config(scrollregion=(0, 0, new_width, new_height))
            
            # Draw existing annotations for this image
            self.draw_existing_annotations()
            
            # Update annotations treeview
            self.update_annotations_tree()
            
            # Update status
            self.status_var.set(f"Loaded image: {self.current_image_name} ({new_width}x{new_height}, Original: {img.width}x{img.height}) - Scale: {self.scale_factor:.2f}")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load image: {str(e)}")
    
    def draw_existing_annotations(self):
        """Draw rectangles for existing annotations on the current image"""
        if not self.current_image_name:
            return
            
        frame_number = self.extract_frame_number(self.current_image_name)
        for ann in self.coordinates_data:
            if ann["frame"] == frame_number:
                # Scale coordinates according to current display scale
                x1 = ann["x1"] * self.scale_factor
                y1 = ann["y1"] * self.scale_factor
                x2 = ann["x2"] * self.scale_factor
                y2 = ann["y2"] * self.scale_factor
                self.draw_marker(x1, y1, x2, y2, ann["label"])
    
    def extract_frame_number(self, image_path):
        """Extract frame number as integer from image filename"""
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
        
        # Store the starting coordinates of the drag
        self.start_x = self.canvas.canvasx(event.x)
        self.start_y = self.canvas.canvasy(event.y)
        # Create a temporary rectangle (will be updated during drag)
        self.current_rect = self.canvas.create_rectangle(
            self.start_x, self.start_y, self.start_x, self.start_y,
            outline="red", width=2, dash=(4, 4)  # Dashed outline while dragging
        )
    
    def on_canvas_drag(self, event):
        if not self.current_image_path or self.current_rect is None:
            return
        
        # Update the rectangle's end coordinates as the mouse moves
        current_x = self.canvas.canvasx(event.x)
        current_y = self.canvas.canvasy(event.y)
        self.canvas.coords(
            self.current_rect,
            self.start_x, self.start_y, current_x, current_y
        )
        # Update status bar with current rectangle dimensions
        width = abs(current_x - self.start_x) / self.scale_factor
        height = abs(current_y - self.start_y) / self.scale_factor
        self.status_var.set(f"Drawing rectangle: Width={width:.0f}, Height={height:.0f}")
    
    def on_canvas_release(self, event):
        if not self.current_image_path or self.current_rect is None:
            return
        
        # Get the final coordinates
        end_x = self.canvas.canvasx(event.x)
        end_y = self.canvas.canvasy(event.y)
        
        # Normalize coordinates (ensure x1,y1 is top-left and x2,y2 is bottom-right)
        x1 = min(self.start_x, end_x)
        y1 = min(self.start_y, end_y)
        x2 = max(self.start_x, end_x)
        y2 = max(self.start_y, end_y)
        
        # Draw the final rectangle
        tool = self.selected_tool.get()
        self.canvas.delete(self.current_rect)  # Remove temporary dashed rectangle
        self.draw_marker(x1, y1, x2, y2, tool)
        
        # Convert coordinates back to original image scale
        original_x1 = int(x1 / self.scale_factor)
        original_y1 = int(y1 / self.scale_factor)
        original_x2 = int(x2 / self.scale_factor)
        original_y2 = int(y2 / self.scale_factor)
        
        # Extract frame number from image path
        frame_number = self.extract_frame_number(self.current_image_path)
        
        # Save annotation data with rectangle coordinates
        annotation = {
            "frame": frame_number,
            "label": tool,
            "x1": original_x1,
            "y1": original_y1,
            "x2": original_x2,
            "y2": original_y2
        }
        self.coordinates_data.append(annotation)
        
        # Update treeview
        self.tree.insert("", tk.END, values=(
            frame_number,
            tool,
            original_x1,
            original_y1,
            original_x2,
            original_y2
        ))
        
        # Update status
        width = abs(original_x2 - original_x1)
        height = abs(original_y2 - original_y1)
        self.status_var.set(f"Added {tool} rectangle at ({original_x1}, {original_y1}) to ({original_x2}, {original_y2}) - Size: {width}x{height}")
        
        # Reset drag variables
        self.current_rect = None
        self.start_x = None
        self.start_y = None
    
    def draw_marker(self, x1, y1, x2, y2, tool):
        # Draw a rectangle for the annotated region
        self.canvas.create_rectangle(
            x1, y1, x2, y2,
            fill="",  # Transparent fill
            outline="red",
            width=2
        )
        
        # Add a label at the top-left corner of the rectangle
        self.canvas.create_text(
            x1 + 5, y1 + 15,  # Slightly offset from top-left
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
            # Create directory if it doesn't exist
            directory = os.path.dirname(self.json_file_path)
            if directory and not os.path.exists(directory):
                os.makedirs(directory)
                
            # Get current frame number
            frame_number = self.extract_frame_number(self.current_image_path)
                
            # Filter annotations for the current frame
            current_frame_annotations = [
                ann for ann in self.coordinates_data 
                if ann.get("frame") == frame_number
            ]
            
            # Save only annotations for the current frame
            with open(self.json_file_path, 'w') as f:
                json.dump(current_frame_annotations, f, indent=4)
                
            messagebox.showinfo("Success", f"Annotations saved to {self.json_file_path}")
            self.status_var.set(f"Annotations saved to {self.json_file_path}")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to save annotations: {str(e)}")
    
    def update_annotations_tree(self):
        # Clear the tree
        for item in self.tree.get_children():
            self.tree.delete(item)
        
        # Filter annotations for the current image
        frame_number = self.extract_frame_number(self.current_image_name)
        current_image_annotations = [
            ann for ann in self.coordinates_data 
            if ann["frame"] == frame_number
        ]
            
        # Add to tree
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
        
        # Get current frame number
        frame_number = self.extract_frame_number(self.current_image_path)
        
        # Ask for confirmation
        if messagebox.askyesno("Confirm", f"Clear all annotations for frame {frame_number}?"):
            # Remove annotations for current frame
            self.coordinates_data = [
                ann for ann in self.coordinates_data 
                if ann.get("frame") != frame_number
            ]
            
            # Clear annotations from tree
            for item in self.tree.get_children():
                self.tree.delete(item)
            
            # Reload image (to clear markers)
            if self.current_image_path:
                self.load_image(self.current_image_path)
                
            self.status_var.set(f"Cleared annotations for frame {frame_number}")

if __name__ == "__main__":
    root = tk.Tk()
    app = ImageAnnotationTool(root)
    root.mainloop()