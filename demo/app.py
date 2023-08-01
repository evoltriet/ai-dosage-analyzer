# python -m tornado.autoreload hier_predict_app.py

import os
import tornado.ioloop
import tornado.web
import sys
import pandas as pd
import numpy as np
import json
import simplejson
import eli5
try:
    import cPickle as pickle
except ImportError:
    import pickle
try: 
    from BeautifulSoup import BeautifulSoup
except ImportError:
    from bs4 import BeautifulSoup


# Needed to prevent truncation of column widths when storing to html:
pd.set_option('display.max_colwidth', -1)


# Example from: https://stackoverflow.com/questions/18096748/pandas-dataframes-to-html-highlighting-table-rows/34054255
PREDICT_HEADER = '''
<html>
    <head>
        <style>
        </style>
    </head>
    <body>
'''

PREDICT_FOOTER = '''
    </body>
</html>
'''

static_path = 'static/'
port = 8085

class CreatePredictions:
    
    def __init__(self, ambiguous_phrases = None, loaded_vec = None, loaded_qty_model = None,
                 loaded_qty_calibration = None, loaded_qty_names = None, loaded_qty_thresholds = None,
                 loaded_freq_model = None, loaded_freq_calibration = None,
                 loaded_freq_names = None, loaded_freq_thresholds = None):
        #define list of ambiguous phrases that when found, the model will prevent predictions on
        if ambiguous_phrases is None:
            ambiguous_phrases = ["apply", "required", "needed", "prn", "up to", "affected", " or ", "spoon", "weekly", "inject"]
        self.ambiguous_phrases = ambiguous_phrases

        if loaded_vec is None:
            #load vectorizer
            f = open('../pickles/tfidf.pickle', 'rb') 
            loaded_vec = pickle.load(f)     
            f.close()  
        self.loaded_vec = loaded_vec 

        if loaded_qty_model is None:
            #load quantity prediction model
            f = open('../pickles/quantity_prediction_model.sav', 'rb')   
            loaded_qty_model = pickle.load(f)     
            f.close()
        self.loaded_qty_model = loaded_qty_model           
        if loaded_qty_calibration is None:
            #load calibrated quantity model for probability prediction
            f = open('../pickles/calibrated_quantity_model.sav', 'rb')   
            loaded_qty_calibration = pickle.load(f)     
            f.close() 
        self.loaded_qty_calibration = loaded_qty_calibration          
        if loaded_qty_names is None:
            #load quantity target names
            f = open('../pickles/id_to_quant.pkl', 'rb')   
            loaded_qty_names = pickle.load(f)     
            f.close()
        self.loaded_qty_names = loaded_qty_names
        if loaded_qty_thresholds is None:
            #load quantity thresholds
            f = open('../pickles/quantity_thresholds.pkl', 'rb')   
            loaded_qty_thresholds = pickle.load(f)     
            f.close() 
        self.loaded_qty_thresholds = loaded_qty_thresholds  

        if loaded_freq_model is None:
            #load frequency prediction model
            f = open('../pickles/frequency_prediction_model.sav', 'rb')   
            loaded_freq_model = pickle.load(f)     
            f.close()  
        self.loaded_freq_model = loaded_freq_model      
        if loaded_freq_calibration is None:
            #load calibrated frequency model for probability prediction
            f = open('../pickles/calibrated_frequency_model.sav', 'rb')   
            loaded_freq_calibration = pickle.load(f)     
            f.close() 
        self.loaded_freq_calibration = loaded_freq_calibration          
        if loaded_freq_names is None:
            #load frequency target names
            f = open('../pickles/id_to_freq.pkl', 'rb')   
            loaded_freq_names = pickle.load(f)     
            f.close() 
        self.loaded_freq_names = loaded_freq_names
        if loaded_freq_thresholds is None:
            #load frequency thresholds
            f = open('../pickles/frequency_thresholds.pkl', 'rb')   
            loaded_freq_thresholds = pickle.load(f)     
            f.close() 
        self.loaded_freq_thresholds = loaded_freq_thresholds
        
    
    def predict_list(self, note):
        input_grams = self.loaded_vec.transform(note)
        
        # make quantity related predictions
        predicted_class = self.loaded_qty_calibration.predict(input_grams)[0]
        qty_prediction = self.loaded_qty_names[predicted_class]
        pred_prob = pd.DataFrame(self.loaded_qty_calibration.predict_proba(input_grams) * 100, columns=self.loaded_qty_calibration.classes_)
        qty_confidence = pred_prob[predicted_class][0]
        qty_highlight_df = eli5.explain_prediction_df(self.loaded_qty_model, note[0], vec=self.loaded_vec,
                                                      target_names=self.loaded_qty_names, top_targets=1, top=2)
        qty_feat = qty_highlight_df.iloc[0][1]
        qty_pattern = qty_feat.rsplit(' ')[0].upper()
        qty_start = note[0].find(qty_pattern)
        qty_start += 1
        qty_len = len(qty_feat)
        #quantity threshold validation
        if qty_confidence < self.loaded_qty_thresholds[predicted_class]:
            qty_prediction = "Not confident"
            qty_feat = np.nan
            qty_start = np.nan
            qty_len = np.nan
        
        # make frequency related predictions
        predicted_class = self.loaded_freq_calibration.predict(input_grams)[0]
        freq_prediction = self.loaded_freq_names[predicted_class]
        pred_prob = pd.DataFrame(self.loaded_freq_calibration.predict_proba(input_grams) * 100, columns=self.loaded_freq_calibration.classes_)
        freq_confidence = pred_prob[predicted_class][0]
        freq_highlight_df = eli5.explain_prediction_df(self.loaded_freq_model, note[0], vec=self.loaded_vec,
                                                      target_names=self.loaded_freq_names, top_targets=1, top=2)
        freq_feat = freq_highlight_df.iloc[0][1]
        freq_pattern = freq_feat.rsplit(' ')[0].upper()
        freq_start = note[0].find(freq_pattern)
        freq_start += 1
        freq_len = len(freq_feat)
        #frequency threshold validation
        if freq_confidence < self.loaded_freq_thresholds[predicted_class]:
            freq_prediction = "Not confident"
            freq_feat = np.nan
            freq_start = np.nan
            freq_len = np.nan

        #ambiguous phrase filter
        for ambiguous_phrase in self.ambiguous_phrases:
            if ambiguous_phrase in note[0].lower():
                qty_prediction = "Contains ambiguous phrase: "+ambiguous_phrase
                qty_confidence = np.nan
                qty_feat = np.nan
                qty_start = np.nan
                qty_len = np.nan
                freq_prediction = "Contains ambiguous phrase: "+ambiguous_phrase
                freq_confidence = np.nan
                freq_feat = np.nan
                freq_start = np.nan
                freq_len = np.nan
        
        pred_dict = {'qty_prediction': qty_prediction,
                     'qty_confidence': qty_confidence,
                     'qty_feat': qty_feat,
                     'qty_start': qty_start,
                     'qty_len': qty_len,
                     'freq_prediction': freq_prediction,
                     'freq_confidence': freq_confidence,
                     'freq_feat': freq_feat,
                     'freq_start': freq_start,
                     'freq_len': freq_len}
        
        return pred_dict
    
    
    def parse_html(self, highlight_text):
        parsed_html = BeautifulSoup(highlight_text.data, 'html.parser')
        parsed_html.find('p').decompose()
        parsed_html.find('table').decompose()
        for p in parsed_html.find_all('p'):
            if 'style' in p.attrs:
                del p.attrs['style']
        return parsed_html
        

class PredictHandler(tornado.web.RequestHandler):

    def initialize(self, model):
        self.model = model

    def get(self):
        self.write("Hello, world")

    def post(self):

        input_notes = self.get_argument("notes", default=None, strip=False)
        # print(input_notes)
        prediction = self.create_prediction_page(input_notes)

        self.write(prediction)

    def create_prediction_page(self, input_notes):
        # Replace periods with spaces:
        input_notes = input_notes.replace('.', ' ')
        input_notes = input_notes.replace(':', ' ')
        notes_list = input_notes.splitlines()

        # Remove Blank Lines:
        # entry.strip() evaluates to false only if it is an empty string:
        notes_list = [entry for entry in notes_list if entry.strip()]

        print('Predicting for {} notes...'.format(len(notes_list)))

        prediction_html = self.get_html_prediction_from_strings(notes_list=notes_list)

        return PREDICT_HEADER + prediction_html + PREDICT_FOOTER

    def get_html_prediction_from_strings(self, notes_list):

        return pd.DataFrame([{'a': 1, 'b':2}]).to_html(escape=False, max_cols=None, columns=['a', 'b'])


class PredictStaticHandler(tornado.web.RequestHandler):

    def initialize(self, pred_model):
        self.pred_model = pred_model
    def post(self):
        print(self.request.body)
        json_dict = json.loads(self.request.body)

        input_directions = self.json_inputs_to_vector_inputs(json_dict)

        # print(input_notes)
        prediction = self.create_prediction_html(input_directions)

        self.write(prediction)


    def json_inputs_to_vector_inputs(self, json_dict):
        """
        Takes in JSON format data and returns lists of the following features:
        notes, genders, ages, physician_orders"""
        # (Gender == M, Age > 65, PhysicianOrder == Inpatient) = (True, True, True)
        input_notes = [ent['note'] for ent in json_dict]
        # genders = [1 if ent['gender'] == 'Male' else 0 for ent in json_dict]
        # ages = [1 if ent['age'] == '>=65' else 0 for ent in json_dict]
        # pos = [1 if ent['PO'] == 'Inpatient' else 0 for ent in json_dict]

        # static_data = np.vstack([genders, ages, pos]).T

        # return input_notes, genders, ages, pos, static_data

        return input_notes

    def create_prediction_html(self, input_notes, static_data=None):

        qty_notes = []
        qty_preds = []
        qty_probs = []
        qty_starts = []
        qty_lens = []
        freq_notes = []
        freq_preds =[]
        freq_probs = []
        freq_starts = []
        freq_lens = []

        for note in input_notes:
            pred_models = self.pred_model.predict_list([str(note)])
            
            #quantity text highlighting
            qty_highlight_text = eli5.show_prediction(self.pred_model.loaded_qty_model, str(note), vec=self.pred_model.loaded_vec, 
                                                      target_names=self.pred_model.loaded_qty_names, top_targets=1, top=2)
            parsed_html = self.pred_model.parse_html(qty_highlight_text)
            #append quantity results
            qty_notes.append(parsed_html)
            qty_preds.append(pred_models.get('qty_prediction'))
            qty_probs.append(pred_models.get('qty_confidence'))
            qty_starts.append(pred_models.get('qty_start'))
            qty_lens.append(pred_models.get('qty_len'))

            #frequency text highlighting
            freq_highlight_text = eli5.show_prediction(self.pred_model.loaded_freq_model, str(note), vec=self.pred_model.loaded_vec, 
                                                       target_names=self.pred_model.loaded_freq_names, top_targets=1, top=2)
            parsed_html = self.pred_model.parse_html(freq_highlight_text)
            #append frequency results
            freq_notes.append(parsed_html)
            freq_preds.append(pred_models.get('freq_prediction'))
            freq_probs.append(pred_models.get('freq_confidence'))
            freq_starts.append(pred_models.get('freq_start'))
            freq_lens.append(pred_models.get('freq_len'))

        # results_df = pd.DataFrame([{'a': 1, 'b': 2}]).to_html(escape=False, max_cols=None, columns=['a', 'b'])

        colnames = ['Quantity Detections', 
                    'Quantity Predictions', 
                    'Quantity Confidence', 
                    'Quantity Start', 
                    'Quantity Length', 
                    'Frequency Detections', 
                    'Frequency Predictions', 
                    'Frequency Confidence',
                    'Frequency Start',
                    'Frequency Length']
        
        results_df = pd.DataFrame({colnames[0]: qty_notes,
                                   colnames[1]: qty_preds,
                                   colnames[2]: qty_probs,
                                   colnames[3]: qty_starts,
                                   colnames[4]: qty_lens,
                                   colnames[5]: freq_notes,
                                   colnames[6]: freq_preds,
                                   colnames[7]: freq_probs,
                                   colnames[8]: freq_starts,
                                   colnames[9]: freq_lens}).to_html(escape=False, max_cols=None, columns=colnames)

        return results_df

    # def remove_entries_with_no_known_words(self, notes_list, static_data):
    #
    #     # A list which stores the length of each note, after we keep only words which are in the word lookup:
    #     conv_note_lens = []
    #     for note in notes_list:
    #         conv_note_lens.append(len([word2index[word.lower()] for word in note.split() if word.lower() in word2index]))
    #
    #     # now keep only the notes with a length greater than 0:
    #     notes_list = [note for note, length in zip(notes_list, conv_note_lens) if length>0]
    #
    #     # now keep only the static_data rows for notes with a length greater than 0:
    #     static_data = static_data[np.where(np.array(conv_note_lens) > 0), :][0]
    #
    #     return notes_list, static_data

    # def get_html_prediction_from_strings(self, notes_list, static_data=None):
    #     # esb, clean_note_seqs = utils.get_example_sequence_builder(notes_list, word2index_path=word2index_path, static_data=static_data)
    #     preds, attns, clean_note_ind_seqs = utils.predict_example_attentions(model_att,
    #                                                                          notes_list,
    #                                                                          word2index_path,
    #                                                                          static_data=static_data)
    #     examp_df = utils.examp_df_to_vis(preds,
    #                                      clean_note_ind_seqs,
    #                                      notes_list,
    #                                      index2word_path,
    #                                      static_data=static_data)
    #     attn_note = utils.add_attention_column_to_predictions(examp_df,
    #                                                           attns,
    #                                                           ehrid_per_attention=range(0, len(notes_list)),
    #                                                           mono=True)
    #
    #     return attn_note.to_html(escape=False, max_cols=None,
    #                              columns=['Gender=M', 'Age>=65', 'PO=Inpatient', #'note_orig', #'note_clean',
    #                                       'note_attn', 'attn_max', 'len', 'score'],
    #                              classes=['table', 'table-hover'])  # tables for bootstrap

    def colour_word_intensity(self, word, intensity=0, max_alpha=0.6):
        # Green:
        # int_word = '<span style="background-color:rgba(0,255,0,'+str(intensity/max_alpha)+');">'+word+'</span>'

        # Orange:
        int_word = '<span style="background-color:rgba(255,153,0,' + str(
            intensity / max_alpha) + ');">' + word + '</span>'

        return int_word

    def colour_word_intensities(self, sentence_or_wordlist, weights, min_threshold=0.1, max_alpha=0.6, mono=True):
        if isinstance(sentence_or_wordlist, str):
            wordlist = sentence_or_wordlist.split()
        elif isinstance(sentence_or_wordlist, list):
            wordlist = sentence_or_wordlist
        # Trim the sentence to be at most as long as the attention vector,
        # necessary as the weights are limited to a length but the sentences
        # are not
        wordlist = wordlist[0:len(weights)]

        assert (len(wordlist) <= len(weights))
        # Handle unicode issues:
        #     wordlist = [unicode(word) for word in wordlist]

        formatted_wordlist = []

        for word, intensity in zip(wordlist, weights):
            # include the word without the intensity styling tag
            # to eliminate unnecessary tabs
            if intensity > min_threshold:
                if mono:
                    formatted_wordlist.append(self.colour_word_intensity(word, intensity, max_alpha=max_alpha))
                else:
                    formatted_wordlist.append(self.colour_word_intensity_all_colours(word, intensity, max_alpha=max_alpha))
            else:
                formatted_wordlist.append(word)

        return ' '.join(formatted_wordlist)
    
class PredictJSONHandler(tornado.web.RequestHandler):

    def initialize(self, pred_model):
        self.pred_model = pred_model

    def post(self):
        print(self.request.body)
        json_dict = json.loads(self.request.body)

        input_directions = self.json_inputs_to_vector_inputs(json_dict)

        # print(input_notes)
        prediction = self.create_prediction_json(input_directions)

        self.write(prediction)


    def json_inputs_to_vector_inputs(self, json_dict):
        """
        Takes in JSON format data and returns lists of the following features:
        notes, genders, ages, physician_orders"""
        # (Gender == M, Age > 65, PhysicianOrder == Inpatient) = (True, True, True)
        input_notes = [ent['note'] for ent in json_dict]
        # genders = [1 if ent['gender'] == 'Male' else 0 for ent in json_dict]
        # ages = [1 if ent['age'] == '>=65' else 0 for ent in json_dict]
        # pos = [1 if ent['PO'] == 'Inpatient' else 0 for ent in json_dict]

        # static_data = np.vstack([genders, ages, pos]).T

        # return input_notes, genders, ages, pos, static_data

        return input_notes

    def create_prediction_json(self, input_notes, static_data=None):

        qty_preds = []
        qty_probs = []
        qty_feats = []
        qty_starts = []
        qty_lens = []
        freq_preds =[]
        freq_probs = []
        freq_feats = []
        freq_starts = []
        freq_lens = []
        json_dict = []

        for note in input_notes:
            pred_models = self.pred_model.predict_list([str(note)])

            #append quantity results
            qty_preds.append(pred_models.get('qty_prediction'))
            qty_probs.append(pred_models.get('qty_confidence'))
            qty_feats.append(pred_models.get('qty_feat'))
            qty_starts.append(pred_models.get('qty_start'))
            qty_lens.append(pred_models.get('qty_len'))

            #append frequency results
            freq_preds.append(pred_models.get('freq_prediction'))
            freq_probs.append(pred_models.get('freq_confidence'))
            freq_feats.append(pred_models.get('freq_feat'))
            freq_starts.append(pred_models.get('freq_start'))
            freq_lens.append(pred_models.get('freq_len'))

        colnames = ['Dosage Text',
                    'Quantity Feature', 
                    'Quantity Start',
                    'Quantity Length',
                    'Quantity Prediction', 
                    'Quantity Confidence',
                    'Frequency Feature',
                    'Frequency Start',
                    'Frequency Length', 
                    'Frequency Prediction', 
                    'Frequency Confidence']
        
        for i in range(0, len(input_notes)):
            json_dict.append(dict(([colnames[0], input_notes[i]],
                                   [colnames[1], qty_feats[i]],
                                   [colnames[2], qty_starts[i]],
                                   [colnames[3], qty_lens[i]],
                                   [colnames[4], qty_preds[i]],
                                   [colnames[5], qty_probs[i]],
                                   [colnames[6], freq_feats[i]],
                                   [colnames[7], freq_starts[i]],
                                   [colnames[8], freq_lens[i]],
                                   [colnames[9], freq_preds[i]],
                                   [colnames[10], freq_probs[i]]
                                   )))

        return simplejson.dumps(json_dict, indent=4, ignore_nan=True)

pred_model = CreatePredictions()
application = tornado.web.Application([
    (r"/predict", PredictHandler, dict(model={})),  # pass the model into the PredictHandler
    (r"/predict_static", PredictStaticHandler, dict(pred_model = pred_model)),
    (r"/predict_json", PredictJSONHandler, dict(pred_model = pred_model)),
    (r"/(.*)", tornado.web.StaticFileHandler, {"path": static_path, "default_filename": "index.html"})
], autoreload=True)

if __name__ == '__main__':
    application.listen(port)
    tornado.ioloop.IOLoop.instance().start()
