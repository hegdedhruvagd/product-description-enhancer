def test_the_page_file_is_valid_python():
    import py_compile
    py_compile.compile("app/main.py", doraise=True)
