from liveFeed import is_valid_label

def test_valid_labels():
    assert is_valid_label("person") == True
    assert is_valid_label("Dalek") == True
    assert is_valid_label("jedi lightsaber") == True
    assert is_valid_label("car") == False
