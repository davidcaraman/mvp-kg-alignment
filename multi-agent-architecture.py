import pandas as pd
import requests
import json
import time
import logging
import os
import ast # Pentru a evalua șirurile care reprezintă liste/dicționare
from tqdm import tqdm # Pentru o bară de progres vizuală
from enum import IntEnum # Pentru a defini ConfidenceLevel
from typing import List, Optional, Dict, Any
import concurrent.futures # Nou: pentru execuție concurentă

# Importuri Pydantic
from pydantic import BaseModel, Field, ConfigDict, ValidationError

# --- Configurare Jurnalizare (Logging) ---
# Configurăm jurnalizarea pentru toți agenții.
# Mesajele vor include ora, numele agentului și nivelul de severitate.
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
orchestrator_logger = logging.getLogger("OrchestratorAgent")
classifier_logger = logging.getLogger("ClassifierAgent")
ollama_logger = logging.getLogger("OllamaAPI")

# --- Definiții de tip pentru structurile de date ---
class ConfidenceLevel(IntEnum):
    """Niveluri de încredere pentru alinierea entităților."""
    LOW = 1
    MEDIUM_LOW = 2
    MEDIUM = 3
    MEDIUM_HIGH = 4
    HIGH = 5

class EntityAlignment(BaseModel):
    """Model pentru alinierea unei entități."""
    batch_id: str = Field(..., description="ID-ul lotului de procesare din care face parte entitatea.") # Adăugat batch_id
    kg1_entity_id: str = Field(..., description="ID-ul entității din KG1")
    kg2_entity_id: Optional[str] = Field(None, description="ID-ul celei mai bune potriviri din KG2 sau null")
    confidence_score: ConfidenceLevel = Field(..., description="Nivelul de încredere 1-5")
    explanation: str = Field(..., description="Explicație detaliată în română pentru decizie")
    evidence: List[str] = Field(default_factory=list, description="Lista cu dovezile identificate")
    
    # Flag-uri pentru agenți suplimentari
    need_judge: bool = Field(False, description="Necesită decizie finală")
    
    # Metadata pentru tracking
    processing_time_ms: Optional[float] = Field(None, description="Timpul de procesare în milisecunde")
    
    model_config = ConfigDict(
        use_enum_values=True, # Exportă valorile numerice ale enum-urilor
        json_schema_extra={
            "example": {
                "batch_id": "20240620-001",
                "kg1_entity_id": "zh:Barack_Obama",
                "kg2_entity_id": "en:Barack_Obama", 
                "confidence_score": 5,
                "explanation": "Am identificat o potrivire perfectă între '巴拉克·奥巴马' și 'Barack Obama' pe baza următoarelor criterii: ambele sunt de tip Person, au aceeași dată de naștere (4 august 1961), au ocupat funcția de președinte al SUA în aceeași perioadă (2009-2017).",
                "evidence": [
                    "Ambele entități sunt de tip Person",
                    "Data nașterii identică: 4 august 1961", 
                    "Ocupația identică: Președinte SUA",
                    "Perioada în funcție identică: 2009-2017"
                ],
                "need_judge": False,
                "processing_time_ms": 500.25
            }
        }
    )

# --- 1. Agentul Orchestrator ---
class OrchestratorAgent:
    """
    Agentul Orchestrator este responsabil pentru inițierea fluxului de lucru,
    delegarea sarcinilor de clasificare către Agentul Clasificator și coordonarea
    trimiterii seturilor de date către agenții specializați.
    """
    def __init__(self, 
                 data_source_path="processed_data/entity_alignment_with_names_and_uris.csv",
                 use_pre_classified_data: bool = False, # Nou: să încarce date clasificate anterior
                 pre_classified_data_dir: Optional[str] = None): # Modificat: calea către DIRECTORUL cu fișierele de date clasificate
        
        self.data_source_path = data_source_path
        self.use_pre_classified_data = use_pre_classified_data
        self.pre_classified_data_dir = pre_classified_data_dir

        # O hartă a tuturor entităților sursă cu candidații lor, pentru căutare rapidă
        self.all_source_entities: Dict[int, Dict] = self._load_all_source_entities()
        # O hartă a entităților pre-clasificate {index_id: category}
        self.pre_classified_categories: Dict[int, str] = {}
        # Un set de ID-uri care au fost deja procesate în această rulare (pentru a evita duplicarea)
        self.classified_ids_this_run: set = set() 

        self.classification_agent = ClassifierAgent()     
        
        self.classified_data_buffer = {
            "Person": [], "Place": [], "Event": [], 
            "building/place": [], "Creative Work": [], "uncertain": [],
            "place": [], "creative_work": [], "event": [], "buildings/places": []
        }
        self.category_threshold = 20 
        self.specialized_agents = self._initialize_specialized_agents() 
        self.batch_counter = 0 
        orchestrator_logger.info("Orchestrator: Agentul a fost inițializat.")

        # Nou: Executor pentru procesarea concurentă a loturilor de către agenții specializați
        # Numărul de lucrători (threads) poate fi ajustat în funcție de resursele disponibile
        self.executor = concurrent.futures.ThreadPoolExecutor(max_workers=os.cpu_count() * 2)
        self.active_futures: List[concurrent.futures.Future] = [] # Pentru a urmări sarcinile concurente

        # Dacă este activat și un director este specificat, încarcă datele pre-clasificate
        if self.use_pre_classified_data and self.pre_classified_data_dir:
            self._load_pre_classified_categories(self.pre_classified_data_dir)

    def _load_all_source_entities(self) -> Dict[int, Dict]:
        """
        Încarcă toate entitățile sursă cu candidații lor într-o hartă pentru căutare rapidă.
        Returnează un dicționar {index_id: {entity_details_with_candidates}}.
        """
        orchestrator_logger.info(f"Orchestrator: Încărcare toate entitățile sursă din {self.data_source_path}...")
        try:
            # Verifică dacă data_source_path este un director sau un fișier
            if os.path.isdir(self.data_source_path):
                # Dacă este director, încarcă toate fișierele CSV din el
                orchestrator_logger.info(f"Orchestrator: Directorul '{self.data_source_path}' detectat. Încărcare toate fișierele CSV...")
                all_dfs = []
                csv_files_found = [f for f in os.listdir(self.data_source_path) if f.endswith('.csv')]
                orchestrator_logger.info(f"Orchestrator: Găsite {len(csv_files_found)} fișiere CSV în directorul '{self.data_source_path}': {csv_files_found[:10]}...")  # Show first 10 files
                
                for filename in csv_files_found:
                    file_path = os.path.join(self.data_source_path, filename)
                    try:
                        df_file = pd.read_csv(file_path)
                        all_dfs.append(df_file)
                        orchestrator_logger.info(f"Orchestrator: Încărcat {filename} ({len(df_file)} entități). Coloane: {list(df_file.columns)}")
                    except Exception as e:
                        orchestrator_logger.error(f"Orchestrator: Eroare la încărcarea fișierului {filename}: {e}")
                
                if not all_dfs:
                    orchestrator_logger.error(f"Orchestrator: Nu au fost găsite fișiere CSV valide în directorul '{self.data_source_path}'.")
                    return {}
                
                # Combină toate dataframe-urile
                df = pd.concat(all_dfs, ignore_index=True)
                orchestrator_logger.info(f"Orchestrator: Combinat {len(all_dfs)} fișiere CSV în total {len(df)} entități.")
            else:
                # Dacă este fișier, încarcă-l direct
                df = pd.read_csv(self.data_source_path)
            
            # Convertim șirul 'candidates_str' înapoi într-o listă de liste
            if 'candidates_str' in df.columns:
                df['candidates'] = df['candidates_str'].apply(ast.literal_eval)
            else:
                orchestrator_logger.error(f"Orchestrator: Coloana 'candidates_str' nu a fost găsită în fișierele CSV. Coloanele disponibile: {list(df.columns)}")
                return {}
            
            entities_map = {}
            for _, row in df.iterrows():
                entity_id = int(row['index_id'])
                entities_map[entity_id] = {
                    'index_name': row['index_name'],
                    'index_uri': row['index_uri'],
                    'index_id': entity_id,
                    'index_language': row['index_language'],
                    'candidates': row['candidates']
                }
            orchestrator_logger.info(f"Orchestrator: {len(entities_map)} entități sursă încărcate.")
            return entities_map
        except FileNotFoundError:
            orchestrator_logger.error(f"Orchestrator: Fișierul sursă '{self.data_source_path}' nu a fost găsit. Asigură-te că preprocesarea a fost rulată pentru a genera 'entity_alignment_with_names_and_uris.csv'.")
            return {}
        except Exception as e:
            orchestrator_logger.error(f"Orchestrator: Eroare la încărcarea entităților sursă: {e}")
            return {}

    def _load_pre_classified_categories(self, directory_path: str):
        """
        Încarcă categorii pre-clasificate din toate fișierele CSV dintr-un director dat.
        Populează self.pre_classified_categories cu {index_id: category}.
        """
        orchestrator_logger.info(f"Orchestrator: Încărcare categorii pre-clasificate din directorul '{directory_path}'...")
        if not os.path.isdir(directory_path):
            orchestrator_logger.warning(f"Orchestrator: Directorul '{directory_path}' nu a fost găsit. Ignoră pre-încărcarea categoriilor.")
            return

        for filename in os.listdir(directory_path):
            if filename.endswith('.csv'):
                file_path = os.path.join(directory_path, filename)
                try:
                    df_pre_classified = pd.read_csv(file_path)
                    # Ne așteptăm la coloanele 'index_id' și 'category' conform cerinței tale
                    if 'index_id' in df_pre_classified.columns and 'category' in df_pre_classified.columns:
                        for _, row in df_pre_classified.iterrows():
                            entity_id = int(row['index_id'])
                            category = str(row['category']) # Folosim 'category' direct
                            self.pre_classified_categories[entity_id] = category
                        orchestrator_logger.info(f"Orchestrator: {len(df_pre_classified)} entități încărcate din '{filename}'.")
                    else:
                        orchestrator_logger.warning(f"Orchestrator: Fișierul '{filename}' nu conține coloanele 'index_id' și 'category' necesare. Săriți peste.")
                except Exception as e:
                    orchestrator_logger.error(f"Orchestrator: Eroare la încărcarea fișierului pre-clasificat '{file_path}': {e}")
        orchestrator_logger.info(f"Orchestrator: Total {len(self.pre_classified_categories)} entități pre-clasificate încărcate din directorul '{directory_path}'.")


    def _initialize_specialized_agents(self):
        """
        Inițializează instanțele pentru fiecare agent specializat.
        """
        agents = {
            "Person": PersonSpecializedAgent(),
            "Place": PlaceSpecializedAgent(),
            "Event": EventSpecializedAgent(),
            "building/place": BuildingPlaceSpecializedAgent(),
            "Creative Work": CreativeWorkSpecializedAgent(),
            "uncertain": UncertainSpecializedAgent(),
            # Mapări pentru categoriile din fișierele CSV
            "place": PlaceSpecializedAgent(),
            "creative_work": CreativeWorkSpecializedAgent(),
            "event": EventSpecializedAgent(),
            "buildings/places": BuildingPlaceSpecializedAgent()
        }
        return agents

    def start_workflow(self, skip_new_classification: bool = False): 
        """
        Inițiază fluxul de lucru.
        `skip_new_classification`: Dacă True, nu va apela ClassifierAgent pentru entitățile neclasificate, ci se va baza doar pe date pre-clasificate.
        """
        if not self.all_source_entities:
            orchestrator_logger.warning("Orchestrator: Nu există entități sursă de procesat. Oprește fluxul de lucru.")
            return

        orchestrator_logger.info("Orchestrator: Pornire flux de lucru de clasificare a entităților.")
        # Curățăm fișierele _aligned.csv la începutul fiecărei rulări complete pentru a evita datele vechi
        self._clean_previous_aligned_files()

        # Faza 1: Procesează entitățile pre-clasificate (dacă sunt disponibile și activat)
        if self.use_pre_classified_data and self.pre_classified_categories:
            orchestrator_logger.info("Orchestrator: Procesare entități pre-clasificate...")
            
            # Procesăm doar ID-urile pre-clasificate care există și în datele sursă
            pre_classified_ids_to_process = [
                eid for eid in self.pre_classified_categories.keys() if eid in self.all_source_entities
            ]
            
            for entity_id in tqdm(pre_classified_ids_to_process, desc="Orchestrator: Procesare pre-clasificate"):
                # Asigurăm că nu procesăm aceeași entitate de două ori în aceeași rulare
                if entity_id in self.classified_ids_this_run: 
                    continue

                full_entity_data = self.all_source_entities[entity_id]
                category = self.pre_classified_categories[entity_id]

                # Construim un dicționar 'classified_entity' similar cu cel returnat de ClassifierAgent
                classified_entity = {
                    'entity_name': full_entity_data['index_name'],
                    'entity_uri': full_entity_data['index_uri'],
                    'entity_id': full_entity_data['index_id'],
                    'entity_language': full_entity_data['index_language'],
                    'entity_category': category,
                    'candidates': full_entity_data['candidates'], # Candidatii sunt esențiali pentru agenții specializați
                    'timestamp': time.strftime('%Y-%m-%d %H:%M:%S') 
                }
                
                self._add_to_buffer_and_send(classified_entity)
                self.classified_ids_this_run.add(entity_id) # Marcăm ID-ul ca procesat

        # Faza 2: Procesează entitățile noi/rămase cu Agentul Clasificator, dacă nu este setat să sară peste
        if not skip_new_classification:
            orchestrator_logger.info("Orchestrator: Procesare entități noi/rămase cu Agentul Clasificator...")
            # Iterăm prin TOATE entitățile sursă
            for entity_id, entity_data in tqdm(self.all_source_entities.items(), desc="Orchestrator: Delegare clasificare nouă"):
                # Verificăm dacă entitatea a fost deja procesată (fie pre-clasificată, fie deja trimisă clasificatorului)
                if entity_id in self.classified_ids_this_run:
                    orchestrator_logger.debug(f"Orchestrator: Salt peste entitatea {entity_id} - deja procesată în această rulare.")
                    continue

                # Dacă entitatea nu a fost procesată, trimite-o clasificatorului
                classified_entity = self.classification_agent.classify_entity(
                    entity_data['index_name'],
                    entity_data['index_uri'],
                    entity_data['index_id'],
                    entity_data['index_language'],
                    entity_data['candidates']
                )
                self._add_to_buffer_and_send(classified_entity)
                self.classified_ids_this_run.add(entity_id) # Marcăm ID-ul ca procesat

        # Procesează orice date rămase în buffer la sfârșit
        self.process_remaining_data()
        orchestrator_logger.info("Orchestrator: Flux de lucru de clasificare finalizat.")
        
        # Așteaptă finalizarea tuturor sarcinilor trimise executorului
        orchestrator_logger.info("Orchestrator: Așteptare finalizare sarcini agenți specializați...")
        for future in tqdm(concurrent.futures.as_completed(self.active_futures), total=len(self.active_futures), desc="Orchestrator: Așteptare agenți specializați"):
            try:
                future.result() # Obține rezultatul (sau re-ridică excepția dacă a apărut una)
            except Exception as exc:
                orchestrator_logger.error(f"Specialized agent task generated an exception: {exc}")
        self.executor.shutdown(wait=True) # Închide executorul după ce toate sarcinile s-au terminat

        # Apelăm funcția de consolidare la sfârșitul fluxului de lucru
        self.consolidate_aligned_results()

    def _add_to_buffer_and_send(self, classified_entity: Dict):
        """
        Funcție helper pentru a adăuga entitatea în buffer-ul categoriei și a verifica pragul.
        """
        category = classified_entity['entity_category']
        if category in self.classified_data_buffer:
            self.classified_data_buffer[category].append(classified_entity)
        else:
            orchestrator_logger.warning(f"Orchestrator: Categorie necunoscută '{category}' detectată. Adăugare la buffer.")
            self.classified_data_buffer[category] = [classified_entity]
        
        orchestrator_logger.debug(f"Orchestrator: Entitate '{classified_entity['entity_name']}' clasificată ca '{category}'. Buffer curent '{category}': {len(self.classified_data_buffer[category])}")

        if len(self.classified_data_buffer[category]) >= self.category_threshold:
            orchestrator_logger.info(f"Orchestrator: Categoria '{category}' a atins pragul de {self.category_threshold} entități.")
            self.batch_counter += 1
            current_batch_id = f"{time.strftime('%Y%m%d')}-{self.batch_counter:03d}"
            self._send_to_specialized_agent(category, self.classified_data_buffer[category], current_batch_id)
            self.classified_data_buffer[category] = []

    def _send_to_specialized_agent(self, category: str, data_list: List[Dict], batch_id: str):
        """
        Trimite setul de date către agentul specializat corespunzător, incluzând batch_id.
        Sarcina este trimisă executorului pentru a rula concurent.
        """
        if category in self.specialized_agents:
            orchestrator_logger.info(f"Orchestrator: Trimitere {len(data_list)} entități din categoria '{category}' (Batch ID: {batch_id}) către Agentul Specializat pe {category} (rulare concurentă).")
            # Submit the task to the executor
            future = self.executor.submit(self.specialized_agents[category].process_data, data_list, batch_id)
            self.active_futures.append(future)
        else:
            orchestrator_logger.error(f"Orchestrator: Nu s-a găsit agent specializat pentru categoria '{category}'.")

    def process_remaining_data(self):
        """
        Procesează orice date rămase în buffer la finalul iterației principale.
        """
        orchestrator_logger.info("Orchestrator: Procesare date rămase în buffer...")
        for category, data_list in self.classified_data_buffer.items():
            if data_list: 
                self.batch_counter += 1
                current_batch_id = f"{time.strftime('%Y%m%d')}-{self.batch_counter:03d}"
                orchestrator_logger.info(f"Orchestrator: Trimitere {len(data_list)} entități rămase din categoria '{category}' (Batch ID: {current_batch_id}) către Agentul Specializat pe {category}.")
                self._send_to_specialized_agent(category, data_list, current_batch_id)

    def _clean_previous_aligned_files(self):
        """
        Șterge fișierele CSV _aligned.csv existente și fișierul consolidat la începutul unei noi rulări.
        Acest lucru previne acumularea de date vechi.
        """
        orchestrator_logger.info("Orchestrator: Curățare fișiere CSV _aligned.csv existente și fișier consolidat...")
        for filename in os.listdir('processed_data'):
            if filename.endswith('_aligned.csv') or filename == 'all_aligned_entities.csv':
                file_path = os.path.join('processed_data', filename)
                try:
                    os.remove(file_path)
                    orchestrator_logger.info(f"Orchestrator: Șters {file_path}")
                except Exception as e:
                    orchestrator_logger.error(f"Orchestrator: Eroare la ștergerea fișierului {file_path}: {e}")
        orchestrator_logger.info("Orchestrare: Curățare fișiere finalizată.")

    def consolidate_aligned_results(self):
        """
        Consolidează toate fișierele CSV care se termină cu '_aligned.csv' într-un singur fișier.
        """
        orchestrator_logger.info("Orchestrator: Inițiere consolidare a tuturor fișierelor _aligned.csv.")
        all_aligned_dfs = []
        output_consolidated_file = os.path.join('processed_data', 'all_aligned_entities.csv')
        
        for filename in os.listdir('processed_data'):
            if filename.endswith('_aligned.csv'):
                file_path = os.path.join('processed_data', filename)
                try:
                    df_part = pd.read_csv(file_path)
                    all_aligned_dfs.append(df_part)
                    orchestrator_logger.info(f"Orchestrator: Încărcat {file_path} ({len(df_part)} rânduri).")
                except Exception as e:
                    orchestrator_logger.error(f"Orchestrator: Eroare la încărcarea fișierului {file_path} pentru consolidare: {e}")
        
        if all_aligned_dfs:
            final_consolidated_df = pd.concat(all_aligned_dfs, ignore_index=True)
            final_consolidated_df.to_csv(output_consolidated_file, index=False)
            orchestrator_logger.info(f"Orchestrator: Toate rezultatele aliniate au fost consolidate în {output_consolidated_file} ({len(final_consolidated_df)} rânduri totale).")
        else:
            orchestrator_logger.warning("Orchestrator: Nu au fost găsite fișiere _aligned.csv pentru consolidare.")

# --- 2. Agentul Clasificator ---
class ClassifierAgent:
    """
    Agentul Clasificator este responsabil pentru clasificarea unei singure entități
    folosind un model LLM (gemma:4b prin Ollama).
    """
    def __init__(self, model_name="gemma3:4b", ollama_url="http://localhost:11434/api/generate"):
        self.model_name = model_name
        self.ollama_url = ollama_url
        self.categories = ["Person", "Place", "Event", "building/place", "Creative Work", "uncertain"]
        classifier_logger.info("ClassifierAgent: Inițializat.")
        self._check_ollama_connection() # Verifică conexiunea la inițializare

    def _check_ollama_connection(self):
        """Verifică dacă Ollama rulează și modelul gemma:4b este disponibil."""
        try:
            url = "http://localhost:11434/api/tags"
            response = requests.get(url, timeout=5)
            response.raise_for_status() 
            
            models = response.json().get('models', [])
            gemma_available = any(self.model_name in model.get('name', '') for model in models)
            
            if gemma_available:
                classifier_logger.info(f"ClassifierAgent: ✅ Ollama rulează și modelul '{self.model_name}' este disponibil.")
                return True
            else:
                classifier_logger.error(f"ClassifierAgent: ❌ Modelul '{self.model_name}' nu a fost găsit. Modele disponibile: {[model.get('name', 'Necunoscut') for model in models]}")
                return False
                
        except requests.exceptions.RequestException as e:
            classifier_logger.error(f"ClassifierAgent: ❌ Eșec la conectarea la Ollama: {str(e)}. Asigură-te că Ollama rulează pe localhost:11434.")
            return False

    def _query_ollama_gemma(self, entity_name, entity_uri, max_retries=3, delay=1):
        """
        Interoghează modelul Ollama gemma:4b pentru a categoriza o entitate.
        Include logică de reîncercare și curățare a răspunsului.
        """
        prompt = f"""Ești un expert clasificator de entități. Având un nume de entitate și URI-ul său, clasifică-l într-una dintre aceste categorii:
- Person: Persoane individuale, figuri istorice, celebrități, etc.
- Place: Locații geografice, orașe, țări, regiuni, etc.
- Event: Evenimente istorice, competiții, festivaluri, războaie, etc.
- building/place: Clădiri specifice, monumente, structuri, locuri de desfășurare, etc.
- Creative Work: Cărți, filme, cântece, opere de artă, publicații, etc.
- uncertain: Când categoria este neclară sau ambiguă

Nume entitate: {entity_name}
URI entitate: {entity_uri}

Bazat pe numele entității și contextul URI, clasifică această entitate. Răspunde DOAR cu una dintre aceste categorii exacte: {', '.join(self.categories)}

Category:"""

        data = {
            "model": self.model_name,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": 0.1,
                "top_p": 0.9,
                "num_predict": 50
            }
        }

        for attempt in range(max_retries):
            try:
                response = requests.post(self.ollama_url, json=data, timeout=30)
                response.raise_for_status() 
                result = response.json()
                category = result.get('response', '').strip()

                category = category.replace('.', '').replace('\\n', '').strip()

                for cat in self.categories:
                    if cat.lower() in category.lower():
                        return cat

                category_lower = category.lower()
                if any(word in category_lower for word in ['person', 'people', 'individual', 'human']):
                    return "Person"
                elif any(word in category_lower for word in ['place', 'location', 'city', 'country', 'region']):
                    return "Place"
                elif any(word in category_lower for word in ['event', 'competition', 'festival', 'war']):
                    return "Event"
                elif any(word in category_lower for word in ['building', 'structure', 'monument', 'venue']):
                    return "building/place"
                elif any(word in category_lower for word in ['creative', 'work', 'book', 'movie', 'song', 'art', 'publication']):
                    return "Creative Work"
                else:
                    return "uncertain"

            except requests.exceptions.RequestException as e:
                classifier_logger.warning(f"ClassifierAgent: Încercarea {attempt + 1} eșuată pentru {entity_name}: {str(e)}")
                if attempt < max_retries - 1:
                    time.sleep(delay * (attempt + 1))
                else:
                    classifier_logger.error(f"ClassifierAgent: Toate încercările au eșuat pentru {entity_name}")
                    return "uncertain"
            except Exception as e:
                classifier_logger.error(f"ClassifierAgent: Eroare neașteptată pentru {entity_name}: {str(e)}")
                return "uncertain"
        return "uncertain" 

    def classify_entity(self, entity_name, entity_uri, entity_id, entity_language, candidates: List[List[Any]]):
        """
        Clasifică o singură entitate folosind LLM și returnează rezultatul complet,
        incluzând acum și lista de candidați originală.
        """
        classifier_logger.info(f"ClassifierAgent: Începe clasificarea pentru: '{entity_name}' (ID: {entity_id})")
        category = self._query_ollama_gemma(entity_name, entity_uri)
        
        result = {
            'entity_name': entity_name,
            'entity_uri': entity_uri,
            'entity_id': entity_id,
            'entity_language': entity_language,
            'entity_category': category,
            'candidates': candidates, 
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
        }
        classifier_logger.info(f"ClassifierAgent: '{entity_name}' clasificată ca: {category}")
        return result

# --- 3. Agenți Specializați pe Categorii (Placeholder-uri) ---
class SpecializedAgent:
    """
    Clasa de bază pentru agenții specializați.
    """
    def __init__(self, category_name: str, alignment_prompt_template: str, ollama_port: int = 11434):
        self.category_name = category_name
        self.agent_logger = logging.getLogger(f"SpecializedAgent[{category_name}]")
        self.agent_logger.info(f"Agent Specializat [{category_name}]: Inițializat cu portul Ollama {ollama_port}.")
        
        # Ollama configuration instead of Gemini
        self.model_name = "gemma3:1b"
        self.ollama_port = ollama_port
        self.ollama_url = f"http://localhost:{ollama_port}/api/generate"
        self.alignment_prompt_template = alignment_prompt_template
        
        # Check Ollama connection on initialization
        self._check_ollama_connection()

    def _check_ollama_connection(self):
        """Verifică dacă Ollama rulează și modelul gemma3:1b este disponibil pe portul specificat."""
        try:
            tags_url = f"http://localhost:{self.ollama_port}/api/tags"
            response = requests.get(tags_url, timeout=5)
            response.raise_for_status() 
            
            models = response.json().get('models', [])
            gemma_available = any(self.model_name in model.get('name', '') for model in models)
            
            if gemma_available:
                self.agent_logger.info(f"SpecializedAgent[{self.category_name}]: ✅ Ollama rulează pe portul {self.ollama_port} și modelul '{self.model_name}' este disponibil.")
                return True
            else:
                self.agent_logger.error(f"SpecializedAgent[{self.category_name}]: ❌ Modelul '{self.model_name}' nu a fost găsit pe portul {self.ollama_port}. Modele disponibile: {[model.get('name', 'Necunoscut') for model in models]}")
                return False
                
        except requests.exceptions.RequestException as e:
            self.agent_logger.error(f"SpecializedAgent[{self.category_name}]: ❌ Eșec la conectarea la Ollama pe portul {self.ollama_port}: {str(e)}. Asigură-te că Ollama rulează pe localhost:{self.ollama_port}.")
            return False

    def _call_ollama_for_alignment(self, index_entity_data: Dict, candidate_entities_data: List[List[Any]], batch_id: str) -> EntityAlignment:
        """
        Interoghează Ollama pentru a efectua alinierea entităților, folosind prompt-ul specific agentului.
        Returnează o instanță EntityAlignment.
        """
        start_time = time.time()

        candidate_list_str = ""
        for i, cand in enumerate(candidate_entities_data):
            cand_name, cand_uri, cand_id, cand_lang = cand
            candidate_list_str += f"{i+1}. Nume: '{cand_name}', URI: '{cand_uri}', ID: {cand_id}, Limbă: {cand_lang}\n"
        
        # Build the full prompt using the category-specific template
        full_prompt = self.alignment_prompt_template.format(
            index_entity_name=index_entity_data['entity_name'],
            index_entity_language=index_entity_data['entity_language'],
            index_entity_uri=index_entity_data['entity_uri'],
            candidates_list=candidate_list_str
        )
        
        # Add JSON format instruction to the prompt
        json_instruction = """

Răspunde DOAR cu un JSON valid în acest format exact:
{
    "kg2_entity_id": "ID-ul candidatului potrivit sau null",
    "confidence_score": număr între 1-5,
    "explanation": "explicația detaliată în română",
    "evidence": ["dovadă1", "dovadă2", "dovadă3"],
    "need_judge": true/false
}"""
        
        full_prompt += json_instruction

        # Ollama API payload
        payload = {
            "model": self.model_name,
            "prompt": full_prompt,
            "stream": False,
            "options": {
                "temperature": 0.1,
                "top_p": 0.9,
                "num_predict": 500
            }
        }

        try:
            response = requests.post(self.ollama_url, json=payload, timeout=200)
            response.raise_for_status()
            
            result = response.json()
            ollama_response_text = result.get('response', '').strip()
            
            # Try to extract JSON from the response
            try:
                # Look for JSON in the response - sometimes Ollama adds extra text
                import re
                json_match = re.search(r'\{.*\}', ollama_response_text, re.DOTALL)
                if json_match:
                    json_str = json_match.group()
                    raw_alignment_data = json.loads(json_str)
                else:
                    # If no JSON found, try parsing the whole response
                    raw_alignment_data = json.loads(ollama_response_text)
                
                # Ensure required fields and add missing ones
                raw_alignment_data['kg1_entity_id'] = str(index_entity_data['entity_id'])
                
                # Validate and convert to EntityAlignment
                alignment_model = EntityAlignment(**raw_alignment_data, batch_id=batch_id)
                
                end_time = time.time()
                alignment_model.processing_time_ms = (end_time - start_time) * 1000
                
                self.agent_logger.info(f"SpecializedAgent[{self.category_name}]: Aliniere procesată pentru {index_entity_data['entity_name']}. Confidență: {alignment_model.confidence_score.value}")
                return alignment_model
                
            except json.JSONDecodeError as e:
                self.agent_logger.error(f"SpecializedAgent[{self.category_name}]: Eroare la parsarea JSON pentru {index_entity_data['entity_name']}: {e}. Răspuns brut: {ollama_response_text}")
                return self._create_default_alignment_result(index_entity_data, f"Eroare la parsarea răspunsului JSON: {e}", batch_id)
                
        except ValidationError as e:
            self.agent_logger.error(f"SpecializedAgent[{self.category_name}]: Eroare de validare Pydantic pentru {index_entity_data['entity_name']}: {e}")
            return self._create_default_alignment_result(index_entity_data, f"Eroare de validare Pydantic: {e}", batch_id)
        except requests.exceptions.RequestException as e:
            self.agent_logger.error(f"SpecializedAgent[{self.category_name}]: Eroare la apelul Ollama API pentru {index_entity_data['entity_name']}: {e}")
            return self._create_default_alignment_result(index_entity_data, f"Eroare la apelul Ollama API: {e}", batch_id)
        except Exception as e:
            self.agent_logger.error(f"SpecializedAgent[{self.category_name}]: Eroare generală la procesarea alinierii pentru {index_entity_data['entity_name']}: {e}")
            return self._create_default_alignment_result(index_entity_data, f"Eroare generală: {e}", batch_id)

    def _create_default_alignment_result(self, index_entity_data: Dict, explanation: str, batch_id: str) -> EntityAlignment:
        """Creează un rezultat de aliniere implicit în caz de eroare sau incertitudine."""
        return EntityAlignment(
            batch_id=batch_id, 
            kg1_entity_id=str(index_entity_data['entity_id']), 
            kg2_entity_id=None,
            confidence_score=ConfidenceLevel.LOW, 
            explanation=f"Nu s-a putut determina o aliniere: {explanation}",
            evidence=["Eroare sau lipsă de date pentru aliniere."],
            need_judge=True, 
            processing_time_ms=None
        )

    def process_data(self, data_list: List[Dict], batch_id: str):
        """
        Metodă generică pentru procesarea datelor.
        Aceasta este acum responsabilă pentru efectuarea alinierii entităților folosind Ollama API.
        """
        self.agent_logger.info(f"Agent Specializat [{self.category_name}]: Am primit {len(data_list)} entități pentru procesare și aliniere (Batch ID: {batch_id}).")
        
        alignment_results: List[EntityAlignment] = []
        for entity_info in data_list:
            index_entity = {
                'entity_name': entity_info['entity_name'],
                'entity_uri': entity_info['entity_uri'],
                'entity_id': entity_info['entity_id'],
                'entity_language': entity_info['entity_language']
            }
            candidates = entity_info['candidates'] 

            alignment_result = self._call_ollama_for_alignment(index_entity, candidates, batch_id) 
            alignment_results.append(alignment_result)

        # Salvarea rezultatelor de aliniere
        output_file_path = f'processed_data/{self.category_name.replace(" ", "_").lower()}_aligned.csv' 
        
        df_results = pd.DataFrame([res.model_dump() for res in alignment_results])
        
        if not os.path.exists(output_file_path):
            df_results.to_csv(output_file_path, index=False, header=True)
            self.agent_logger.info(f"Agent Specializat [{self.category_name}]: Fișier nou de aliniere creat și rezultate salvate în {output_file_path}")
        else:
            df_results.to_csv(output_file_path, index=False, mode='a', header=False)
            self.agent_logger.info(f"Agent Specializat [{self.category_name}]: Rezultate de aliniere adăugate la fișierul existent {output_file_path}")

# Subclase pentru fiecare categorie de agent specializat cu porturi unice
class PersonSpecializedAgent(SpecializedAgent):
    def __init__(self):
        prompt_template = """Ești un expert în alinierea entităților 'Person'.
Analizează entitatea '{index_entity_name}' (limba: {index_entity_language}, URI: {index_entity_uri}) și determină dacă este aceeași persoană din lumea reală cu oricare dintre următoarele entități candidate din KG2:

{candidates_list}

Concentrează-te pe nume, titluri, date de naștere/deces, ocupații și roluri. Dacă găsești o potrivire, oferă cea mai bună potrivire din KG2. Dacă nu există o potrivire clară, setează kg2_entity_id la null. Motivează-ți decizia, oferă dovezi și un scor de încredere (1-5)."""
        super().__init__("Person", prompt_template, ollama_port=11434)

class PlaceSpecializedAgent(SpecializedAgent):
    def __init__(self):
        prompt_template = """Ești un expert în alinierea entităților 'Place'.
Analizează entitatea '{index_entity_name}' (limba: {index_entity_language}, URI: {index_entity_uri}) și determină dacă este aceeași locație geografică din lumea reală cu oricare dintre următoarele entități candidate din KG2:

{candidates_list}

Concentrează-te pe nume, tipuri de locații (oraș, țară, regiune), coordonate geografice (dacă sunt disponibile implicit în URI), și context administrativ/politic. Dacă găsești o potrivire, oferă cea mai bună potrivire din KG2. Dacă nu există o potrivire clară, setează kg2_entity_id la null. Motivează-ți decizia, oferă dovezi și un scor de încredere (1-5)."""
        super().__init__("Place", prompt_template, ollama_port=11435)

class EventSpecializedAgent(SpecializedAgent):
    def __init__(self):
        prompt_template = """Ești un expert în alinierea entităților 'Event'.
Analizează entitatea '{index_entity_name}' (limba: {index_entity_language}, URI: {index_entity_uri}) și determină dacă este același eveniment din lumea reală cu oricare dintre următoarele entități candidate din KG2:

{candidates_list}

Concentrează-te pe nume, date de desfășurare, locații, participanți cheie și descrieri. Dacă găsești o potrivire, oferă cea mai bună potrivire din KG2. Dacă nu există o potrivire clară, setează kg2_entity_id la null. Motivează-ți decizia, oferă dovezi și un scor de încredere (1-5)."""
        super().__init__("Event", prompt_template, ollama_port=11436)

class BuildingPlaceSpecializedAgent(SpecializedAgent):
    def __init__(self):
        prompt_template = """Ești un expert în alinierea entităților 'Building/Place'.
Analizează entitatea '{index_entity_name}' (limba: {index_entity_language}, URI: {index_entity_uri}) și determină dacă este aceeași clădire/structură din lumea reală cu oricare dintre următoarele entități candidate din KG2:

{candidates_list}

Concentrează-te pe nume, tipul de structură, locație (oraș/țară), arhitect (dacă e menționat) și data construcției (dacă e implicită în URI). Dacă găsești o potrivire, oferă cea mai bună potrivire din KG2. Dacă nu există o potrivire clară, setează kg2_entity_id la null. Motivează-ți decizia, oferă dovezi și un scor de încredere (1-5)."""
        super().__init__("building/place", prompt_template, ollama_port=11437)

class CreativeWorkSpecializedAgent(SpecializedAgent):
    def __init__(self):
        prompt_template = """Ești un expert în alinierea entităților 'Creative Work'.
Analizează entitatea '{index_entity_name}' (limba: {index_entity_language}, URI: {index_entity_uri}) și determină dacă este aceeași operă creativă (carte, film, cântec, pictură etc.) din lumea reală cu oricare dintre următoarele entități candidate din KG2:

{candidates_list}

Concentrează-te pe nume, autor/regizor/artist, gen, data publicării/lansării și URI-uri. Dacă găsești o potrivire, oferă cea mai bună potrivire din KG2. Dacă nu există o potrivire clară, setează kg2_entity_id la null. Motivează-ți decizia, oferă dovezi și un scor de încredere (1-5)."""
        super().__init__("Creative Work", prompt_template, ollama_port=11438)

class UncertainSpecializedAgent(SpecializedAgent):
    def __init__(self):
        prompt_template = """Ești un expert în alinierea entităților.
Analizează entitatea '{index_entity_name}' (limba: {index_entity_language}, URI: {index_entity_uri}) și determină dacă este aceeași entitate din lumea reală cu oricare dintre următoarele entități candidate din KG2:

{candidates_list}

Această entitate a fost clasificată inițial ca 'incertă'. Analizează cu atenție numele și URI-urile pentru orice indiciu subtil. Dacă găsești o potrivire, oferă cea mai bună potrivire din KG2. Dacă nu există o potrivire clară, setează kg2_entity_id la null și argumentează de ce a rămas incertă. Motivează-ți decizia, oferă dovezi și un scor de încredere (1-5)."""
        super().__init__("uncertain", prompt_template, ollama_port=11439)

# --- Blocul Principal de Execuție ---
if __name__ == "__main__":
    # Asigură-te că directorul `processed_data` există pentru a salva ieșirile
    os.makedirs('processed_data', exist_ok=True)
    
    orchestrator_logger.info("Main: Pornire proces multi-agent.")

    # 1. Asigură-te că fișierul de date îmbogățit (cu candidați) există.
    # Acesta ar trebui să fie 'entity_alignment_with_names_and_uris.csv'
    # data_file = "processed_data/entity_alignment_with_names_and_uris.csv"
    # if not os.path.exists(data_file):
    #     orchestrator_logger.critical(f"Main: Fișierul '{data_file}' nu a fost găsit. Te rog, rulează mai întâi secțiunile de preprocesare din notebook-ul kg-alignment.ipynb pentru a genera acest fișier (în special cel care creează 'entity_alignment_with_names_and_uris.csv').")
    #     exit("Eroare: Fișier de date lipsă.")

    # Calea către directorul care conține fișierele CSV pre-clasificate.
    # Asigură-te că acest director există și conține fișierele tale.
    pre_classified_entities_directory = r"mvp-kg-alignment\categories" 
    if not os.path.isdir(pre_classified_entities_directory):
        orchestrator_logger.warning(f"Main: Directorul '{pre_classified_entities_directory}' nu a fost găsit. Asigură-te că l-ai creat și conține fișierele tale CSV pre-clasificate.")
        # Poți crea directorul pentru a evita erorile, dar el ar trebui să conțină fișierele tale
        # os.makedirs(pre_classified_entities_directory, exist_ok=True)


    # --- Configurații de rulare (alege una dintre opțiuni) ---

    # Opțiunea 1: Procesare completă (începe cu pre-clasificate, apoi clasifică cele noi)
    # Orchestratorul va încărca entitățile clasificate anterior din directorul 'categories'
    # și le va trimite agenților specializați. Apoi, va clasifica orice entități rămase
    # (care nu erau în fișierele pre-clasificate) și le va trimite și pe acestea.
    # orchestrator = OrchestratorAgent(
    #     data_source_path=data_file, 
    #     use_pre_classified_data=True, 
    #     pre_classified_data_dir=pre_classified_entities_directory
    # )
    # orchestrator.start_workflow(skip_new_classification=False) 
    # orchestrator_logger.info("Main: Rulare completă (cu pre-clasificare din director + clasificare nouă) finalizată.")

    # Opțiunea 2: Doar folosirea datelor pre-clasificate (fără a rula Agentul Clasificator)
    # Orchestratorul va încărca doar entitățile clasificate anterior din directorul 'categories'
    # și le va trimite agenților specializați. Nu va apela Agentul Clasificator pentru nicio entitate. 
    # Utilă pentru re-procesarea alinierii pe date deja clasificate.
    orchestrator = OrchestratorAgent(
        data_source_path=pre_classified_entities_directory,  # Specify the directory containing your CSV files
        use_pre_classified_data=True, 
        pre_classified_data_dir=pre_classified_entities_directory
    )
    orchestrator.start_workflow(skip_new_classification=True) 
    orchestrator_logger.info("Main: Rulare doar cu date pre-clasificate din director finalizată.")

    # Opțiunea 3: Doar clasificare nouă (fără a folosi date pre-clasificate)
    # Orchestratorul va clasifica toate entitățile din 'entity_alignment_with_names_and_uris.csv'
    # de la zero și le va trimite agenților specializați. Nu va încerca să încarce fișiere pre-clasificate.
    # orchestrator = OrchestratorAgent(
    #     data_source_path=data_file, 
    #     use_pre_classified_data=False 
    # )
    # orchestrator.start_workflow(skip_new_classification=False) 
    # orchestrator_logger.info("Main: Rulare doar cu clasificare nouă finalizată.")

    orchestrator_logger.info("Main: Procesul multi-agent a fost finalizat cu succes.")
