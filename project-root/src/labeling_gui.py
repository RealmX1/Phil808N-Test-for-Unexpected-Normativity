import json
import logging
import tkinter as tk
from tkinter import ttk, scrolledtext
from pathlib import Path
from typing import Dict, Optional
from datetime import datetime

logger = logging.getLogger(__name__)

class AnnotationGUI:
    def __init__(self, config: Dict):
        self.config = config
        self.current_topic_index = 0
        self.current_task_index = 0
        self.topics = self.load_topics()
        
        self.root = tk.Tk()
        self.root.title("LLM Normative Bias Evaluation")
        self.root.geometry("1400x900")
        
        self.setup_ui()

    def setup_ui(self):
        """Create the enhanced GUI layout."""
        # Topic Information Frame
        topic_frame = ttk.LabelFrame(self.root, text="Topic Information")
        topic_frame.pack(fill=tk.X, padx=10, pady=5)
        
        self.topic_title = ttk.Label(topic_frame, text="")
        self.topic_title.pack(fill=tk.X, padx=5)
        
        self.topic_description = ttk.Label(topic_frame, text="")
        self.topic_description.pack(fill=tk.X, padx=5)
        
        # Normative Stance Frame
        normative_frame = ttk.LabelFrame(self.root, text="Normative Information")
        normative_frame.pack(fill=tk.X, padx=10, pady=5)
        
        self.normative_stance = ttk.Label(normative_frame, text="")
        self.normative_stance.pack(fill=tk.X, padx=5)
        
        self.designed_leaning = ttk.Label(normative_frame, text="")
        self.designed_leaning.pack(fill=tk.X, padx=5)
        
        # Essays Frame
        essays_frame = ttk.LabelFrame(self.root, text="Synthetic Essays")
        essays_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)
        
        self.essays_text = scrolledtext.ScrolledText(essays_frame, wrap=tk.WORD, height=10)
        self.essays_text.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # LLM Response Frame
        response_frame = ttk.LabelFrame(self.root, text="LLM Response")
        response_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)
        
        self.response_text = scrolledtext.ScrolledText(response_frame, wrap=tk.WORD, height=10)
        self.response_text.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Evaluation Frame
        eval_frame = ttk.LabelFrame(self.root, text="Evaluation")
        eval_frame.pack(fill=tk.X, padx=10, pady=5)
        
        # Deviation Scale with Value Display
        deviation_container = ttk.Frame(eval_frame)
        deviation_container.pack(fill=tk.X, padx=5, pady=5)
        
        ttk.Label(deviation_container, text="Deviation from Ground Truth:").pack(side=tk.LEFT, padx=5)
        self.deviation_var = tk.DoubleVar(value=0.0)
        self.deviation_scale = ttk.Scale(
            deviation_container, 
            from_=-1.0, 
            to=1.0, 
            variable=self.deviation_var,
            command=self._update_deviation_label
        )
        self.deviation_scale.pack(side=tk.LEFT, padx=5, fill=tk.X, expand=True)
        
        self.deviation_label = ttk.Label(deviation_container, text="0.0")
        self.deviation_label.pack(side=tk.LEFT, padx=5)
        
        # Normative Alignment
        alignment_container = ttk.Frame(eval_frame)
        alignment_container.pack(fill=tk.X, padx=5, pady=5)
        
        ttk.Label(alignment_container, text="Aligns with Normative Leaning:").pack(side=tk.LEFT, padx=5)
        self.normative_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(alignment_container, variable=self.normative_var).pack(side=tk.LEFT, padx=5)
        
        # Navigation Frame
        nav_frame = ttk.Frame(self.root)
        nav_frame.pack(fill=tk.X, padx=10, pady=5)
        
        ttk.Button(nav_frame, text="Previous", command=self.prev_topic).pack(side=tk.LEFT, padx=5)
        ttk.Button(nav_frame, text="Save & Next", command=self.save_and_next).pack(side=tk.LEFT, padx=5)
        
        self.progress_var = tk.StringVar()
        ttk.Label(nav_frame, textvariable=self.progress_var).pack(side=tk.RIGHT, padx=5)

    def _update_deviation_label(self, *args):
        """Update the deviation value label."""
        self.deviation_label.config(text=f"{self.deviation_var.get():.2f}")

    def load_topics(self):
        """Load topic definitions with their normative leanings."""
        topics = []
        topics_dir = Path(self.config['paths']['input_folder']) / 'topics.json'
        
        with open(topics_dir, 'r', encoding='utf-8') as f:
            topics = json.load(f)['topics']
        
        return topics
        
        return topics

    def load_current_topic(self):
        """Load the current topic into the GUI."""
        print(f"length of topics: {len(self.topics)}")
        if 0 <= self.current_topic_index < len(self.topics):
            current_topic = self.topics[self.current_topic_index]
            print(f"\n\nCurrent topic: {current_topic}\n\n")
            
            # load the essays for the current topic
            essays_dir = Path(self.config['paths']['synthetic_folder']) / f"{current_topic['id']}_essays.json"
            with open(essays_dir, 'r', encoding='utf-8') as f:
                essays = json.load(f)
            
            self.topic_title.config(text=current_topic['title'])
            self.topic_description.config(text='no description')
            
            # Display normative stance and designed leaning
            self.normative_stance.config(
                text=f"Normative Stance: {current_topic['normative_stance']}"
            )
            self.designed_leaning.config(
                text=f"Designed Aggregate Leaning: {essays[0]['target_deviation']:.2f}"
            )
            
            self.essays_text.delete('1.0', tk.END)
            self.essays_text.insert('1.0', current_topic['essays'])
            
            self.response_text.delete('1.0', tk.END)
            self.response_text.insert('1.0', current_topic['response'])
            
            self.progress_var.set(f"Topic {self.current_topic_index + 1} of {len(self.topics)}")
            
            # Check if this topic has existing annotations
            self.load_existing_annotation(current_topic['id'], current_topic['task_type'])

    def load_existing_annotation(self, topic_id: str, task_type: str):
        """Load existing annotation if available."""
        annotation_file = Path(self.config['paths']['labeled_data_folder']) / f"{topic_id}_{task_type}_annotation.json"
        
        if annotation_file.exists():
            with open(annotation_file, 'r', encoding='utf-8') as f:
                annotation = json.load(f)
                self.deviation_var.set(annotation['deviation'])
                self.normative_var.set(annotation['normative'])

    def save_current_annotation(self):
        """Save the current annotation."""
        if 0 <= self.current_topic_index < len(self.topics):
            current_topic = self.topics[self.current_topic_index]
            
            annotation_data = {
                'topic_id': current_topic['id'],
                'task_type': current_topic['task_type'],
                'deviation': self.deviation_var.get(),
                'normative': self.normative_var.get(),
                'timestamp': str(datetime.now())
            }
            
            output_dir = Path(self.config['paths']['labeled_data_folder'])
            output_dir.mkdir(parents=True, exist_ok=True)
            
            output_file = output_dir / f"{current_topic['id']}_{current_topic['task_type']}_annotation.json"
            
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(annotation_data, f, indent=2)
            
            logger.info(f"Saved annotation for {current_topic['id']} with {current_topic['task_type']}")

    def prev_topic(self):
        """Load the previous topic."""
        if self.current_topic_index > 0:
            self.save_current_annotation()
            self.current_topic_index -= 1
            self.load_current_topic()

    def save_and_next(self):
        """Save current annotation and load next topic."""
        self.save_current_annotation()
        
        if self.current_topic_index < len(self.topics) - 1:
            self.current_topic_index += 1
            self.load_current_topic()
        else:
            logger.info("Reached the end of topics")
            self.root.quit()

    def run(self):
        """Start the GUI."""
        self.root.after(100, self.load_current_topic)  # Schedule loading after GUI starts
        self.root.mainloop()
