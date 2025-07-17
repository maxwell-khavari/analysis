const { chromium } = require('playwright');
const fs = require('fs').promises;
const path = require('path');
const os = require('os');
const readline = require('readline');

const TOTAL_IMAGES = 1000;
const screenshotDir = path.join(os.homedir(), 'Desktop', 'AI_IMG_GPT');

const cardData = [
  { url: 'https://aiplayground-prod1.stanford.edu/c/62fda901-1882-4733-ae31-5a87d13bcc69', label: 'Card for IMG 3' },
  { url: 'https://aiplayground-prod1.stanford.edu/c/215f1460-c6dd-4edd-9d31-998e55017e80', label: 'Card for IMG 2 V2' },
  { url: 'https://aiplayground-prod1.stanford.edu/c/675f5f9a-1812-40f5-89dc-4b0892b5a69b', label: 'Card for IMG 2', exact: true },
  { url: 'https://aiplayground-prod1.stanford.edu/c/00b9e4f4-310a-4ff5-9fce-4265799efb7f', label: 'Card for IMG 1 V2' },
  { url: 'https://aiplayground-prod1.stanford.edu/c/963e79f8-e56c-4db3-b4a5-4940e9166052', label: 'Card for IMG 1', exact: true },
  { url: 'https://aiplayground-prod1.stanford.edu/c/9c5e5132-bd3d-43c0-9b2c-86ff117aae76', label: 'Card for IMG 3 V2' }
];

const rl = readline.createInterface({
  input: process.stdin,
  output: process.stdout
});

function askQuestion(query) {
  return new Promise(resolve => rl.question(query, resolve));
}

async function main() {
  await fs.mkdir(screenshotDir, { recursive: true });

  const browser = await chromium.launch({ headless: false });
  const context = await browser.newContext();
  const page = await context.newPage();

  try {
    await page.goto('https://aiplayground-prod1.stanford.edu/login');
    console.log('Please log in manually in the opened browser window.');
    await askQuestion('Press Enter once you have logged in and are ready to start generating images...');

    let generated = 0;

    while (generated < TOTAL_IMAGES) {
      for (const card of cardData) {
        if (generated >= TOTAL_IMAGES) break;

        try {
          await page.goto(card.url, { timeout: 30000 });
          await page.getByTestId('nav-new-chat-button').click();
          await page.getByRole('button', { name: card.label, exact: card.exact || false }).click();

          await page.waitForTimeout(5000);

          const baseName = `image-${String(generated + 1).padStart(4, '0')}`;
          await page.screenshot({ path: path.join(screenshotDir, `${baseName}.png`) });

          console.log(`✅ Saved ${baseName}`);
          generated++;
        } catch (err) {
          console.error(`❌ Error on ${card.label}: ${err.message}`);
        }
      }
    }

  } catch (fatalError) {
    console.error('💥 Fatal error occurred:', fatalError.message);
  } finally {
    rl.close();
    await context.close();
    await browser.close();
    console.log('🧹 Browser closed.');
  }
}

main().catch(console.error);
