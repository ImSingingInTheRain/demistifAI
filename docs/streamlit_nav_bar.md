# Streamlit Navigation Bar Integration Guide

> Adapted from the [streamlit-navigation-bar project](https://github.com/gabrieltempass/streamlit-navigation-bar).

This guide explains how to embed the `streamlit-navigation-bar` component inside demistifAI Streamlit flows. It summarizes the API, styling hooks, and recommended project structure to keep navigation consistent with upstream guidance.

## Usage

Call `st_navbar` to place a navigation bar at the top of a Streamlit page. If `st.set_page_config` is not used, `st_navbar` must be the first Streamlit command executed on the page and it should only be called once. When `st.set_page_config` is present, call `st_navbar` immediately after it.

```python
from streamlit_navigation_bar import st_navbar

page = st_navbar(["Home", "Page 1", "Page 2"])
```

The function returns the currently selected page (or `None` until a selection is made) so that you can branch into different content blocks.

## Parameters

| Parameter | Type | Description |
|-----------|------|-------------|
| `pages` | `list[str]` | Ordered labels for each page rendered in the navigation bar. |
| `selected` | `str \| None`, optional | Initial page on first render. Accepts names from `pages`, the `logo_page`, or `None`. Defaults to `logo_page` when a logo is supplied, otherwise the first entry in `pages`. If set to `None`, the navbar initializes empty and returns `None` until the user clicks. |
| `logo_path` | `str`, optional | Absolute path to an SVG logo displayed on the left side of the navbar. Defaults to no logo. |
| `logo_page` | `str \| None`, default=`"Home"` | Return value when the logo is clicked. Set to `None` for a non-clickable logo. |
| `urls` | `dict[str, str]`, optional | Maps page names to external URLs (must exist in `pages`). Clicking the entry opens a new browser tab. |
| `styles` | `dict[str, dict[str, str]]`, optional | CSS overrides keyed by HTML tag or pseudo-class (`"nav"`, `"div"`, `"ul"`, `"li"`, `"a"`, `"img"`, `"span"`, `"active"`, `"hover"`). Use CSS properties/values as strings; CSS variables are supported. |
| `options` | `bool \| dict[str, bool]`, default=`True` | Toggle built-in adjustments like `show_menu`, `show_sidebar`, `hide_nav`, `fix_shadow`, and `use_padding`. Pass a single boolean to set them all at once. |
| `adjust` | `bool`, default=`True` | Enables default CSS fixes to ensure the navbar displays correctly. Set to `False` to opt out and provide your own CSS (also disables option-driven adjustments). |
| `key` | `str \| int`, optional | Unique key for the component. Required if multiple navbars of the same type are used. |

### Return value

`st_navbar` returns the selected page name or `None` if the user has not interacted yet. When a logo is present, clicking it returns the `logo_page` value.

## Styling Reference

Understanding the component's DOM makes targeted styling easier. A navbar created with `pages=["Page one", "Page two"]` and an SVG logo produces the following simplified structure:

```html
<nav>
  <div>
    <ul>
      <li>
        <a>
          <img src="svg_logo" />
        </a>
      </li>
      <li>
        <a>
          <span>Page one</span>
        </a>
      </li>
      <li>
        <a>
          <span>Page two</span>
        </a>
      </li>
    </ul>
  </div>
</nav>
```

- Use the `a` tag to style both the logo and page names.
- The `img` tag affects only the logo, and `span` targets the page labels.

### CSS variables

The component accepts theme variables inside the `styles` dictionary, for example:

```python
styles = {
    "nav": {
        "background-color": "var(--primary-color)"
    }
}
```

Supported variables:

- `--primary-color`
- `--background-color`
- `--secondary-background-color`
- `--text-color`
- `--font`

### Default styles

The default implementation applies the following CSS (simplified for readability):

```css
* {
  margin: 0;
  padding: 0;
}
nav {
  align-items: center;
  background-color: var(--secondary-background-color);
  display: flex;
  font-family: var(--font);
  height: 2.875rem;
  justify-content: center;
  padding-left: 2rem;
  padding-right: 2rem;
}
div {
  max-width: 43.75rem;
  width: 100%;
}
ul {
  display: flex;
  justify-content: space-between;
  width: 100%;
}
li {
  align-items: center;
  display: flex;
  list-style: none;
}
a {
  text-decoration: none;
}
img {
  display: flex;
  height: 1.875rem;
}
span {
  color: var(--text-color);
  display: block;
  text-align: center;
}
span:active {
  color: var(--text-color);
  font-weight: bold;
}
span:hover {
  background-color: transparent;
  color: var(--text-color);
}
```

Override any property by supplying new values for the corresponding HTML tag or pseudo-class in `styles`.

### Maximum width

The `div` tag's `max-width` controls spacing between page names. Increase the default `43.75rem` for many or lengthy labels, and reduce it when you have only a few short entries.

## Options

Toggle navbar behavior by passing a dictionary to `options` (or set all at once by passing `True`/`False`):

- `show_menu`: Display Streamlit's menu button.
- `show_sidebar`: Show the sidebar toggle (requires `st.sidebar` usage for proper behavior).
- `hide_nav`: Hide the default Streamlit multipage sidebar navigation widget.
- `fix_shadow`: Always show the expanded sidebar shadow, useful when navbar and sidebar share colors.
- `use_padding`: Add 6rem of top padding to the body (mirrors Streamlit defaults). Turn off to place content directly under the navbar.

## Recommended project structure

For a seamless navigation experience, structure multi-page apps as follows:

```
your_repository/
├── pages/
│   ├── __init__.py
│   ├── home.py
│   ├── page_1.py
│   └── page_2.py
└── app.py
```

`app.py` acts as the entry point, calls `st.set_page_config`, and renders the navbar once:

```python
import streamlit as st
from streamlit_navigation_bar import st_navbar
import pages as pg

st.set_page_config(initial_sidebar_state="collapsed")

page = st_navbar(["Home", "Page 1", "Page 2"])

if page == "Home":
    pg.home()
elif page == "Page 1":
    pg.page_1()
elif page == "Page 2":
    pg.page_2()
```

Inside `pages/__init__.py`, re-export the page functions for convenient imports:

```python
from .home import home
from .page_1 import page_1
from .page_2 import page_2
```

Each page file (for example, `pages/home.py`) simply renders its content:

```python
import streamlit as st

def home():
    st.write("Foo")
```

This approach keeps configuration centralized, reduces duplication, and produces smoother transitions between pages. It does, however, forego Streamlit's native multipage features like distinct URLs and `st.page_link`, so choose the pattern that best fits your app.

