Given the image and the text in <language of the file>, fill each numbered field with a correct answer, in that language and script.
For tables, fill the whole table with correct answers: every row a different realistic record.
Never repeat the same answer.
For plain text, add correct remarks: in the fields, between the lines, in the margins.
Per field: `label` (page text), `chars` (maximum length), `lines`. Never skip a field; never exceed `chars`.

Output `out_NNNN.json`: {"<form_id>": {"<field number>": "<text>", ...}, ...}  (valid UTF-8 JSON, every form of the batch).
