from playwright.sync_api import sync_playwright
import pandas as pd
import time

URL = "https://www.niftyindices.com/reports/historical-data"


def wait_dropdown(page):
    """Wait until the Index dropdown is populated."""
    for _ in range(30):
        opts = page.locator("select").nth(2).locator("option").count()
        if opts > 1:
            return
        page.wait_for_timeout(500)
    raise Exception("Index dropdown never populated.")


def get_options(page):

    dropdown = page.locator("select").nth(2)

    options = dropdown.locator("option")

    rows = []

    for i in range(options.count()):

        value = options.nth(i).get_attribute("value")
        text = options.nth(i).inner_text().strip()

        if (
            text == ""
            or "Select" in text
            or value is None
            or value == ""
        ):
            continue

        rows.append(
            {
                "display_name": text,
                "internal_name": value,
            }
        )

    return rows


with sync_playwright() as p:

    browser = p.chromium.launch(
        headless=False
    )

    page = browser.new_page()

    page.goto(URL)
    print(page.locator("select").count())
    selects = page.locator("select")

    for i in range(selects.count()):
        print(i, selects.nth(i).get_attribute("id"))


    print(page.locator("select").nth(0).evaluate("el => el.outerHTML"))
    page.wait_for_load_state("networkidle")

    time.sleep(5)

    selects = page.locator("select")

    print("Found", selects.count(), "dropdowns")

    # --------------------------
    # Dropdown indexes
    #
    # 0 = Index Type
    # 1 = Sub Index Type
    # 2 = Index
    # --------------------------

    index_type = selects.nth(0)
    sub_type = selects.nth(1)

    page.wait_for_timeout(10000)
    index_type.select_option(label="Equity")
    print(page.locator("#ddlHistoricaltypee").input_value())

    page.wait_for_timeout(1500)

    master = []

    wanted_subtypes = [
        "Sectoral Indices",
        "Thematic Indices",
    ]

    for subtype in wanted_subtypes:

        print("\n===================================")
        print(subtype)
        print("===================================")

        sub_type.select_option(label=subtype)

        wait_dropdown(page)

        rows = get_options(page)

        print(f"Found {len(rows)} indices")

        for r in rows:

            r["sub_index_type"] = subtype

            master.append(r)

            print(
                r["display_name"],
                " --> ",
                r["internal_name"],
            )

    df = (
        pd.DataFrame(master)
        .drop_duplicates()
        .sort_values(
            ["sub_index_type", "display_name"]
        )
    )

    df.to_csv(
        "index_mapping.csv",
        index=False,
    )

    print()

    print("=" * 60)
    print("Saved", len(df), "indices")
    print("=" * 60)

    browser.close()