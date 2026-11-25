// Dialect metadata and language helpers, shared by the page (mirrors dialects.py).

export const DIALECTS = {
  en: { label: 'English', native: 'English', dir: 'ltr', lang: 'en' },
  'ar-MA': { label: 'Moroccan', native: 'الدارجة', dir: 'rtl', lang: 'ar-MA' },
  'ar-EG': { label: 'Egyptian', native: 'المصري', dir: 'rtl', lang: 'ar-EG' },
  'ar-SA': { label: 'Saudi', native: 'السعودي', dir: 'rtl', lang: 'ar-SA' },
};

// Same ranges as dialects.has_arabic (U+0600–U+06FF, U+0750–U+077F).
export function hasArabic(text) {
  return /[\u0600-\u06FF\u0750-\u077F]/.test(text);
}

export function languageOf(text) {
  return hasArabic(text) ? 'ar' : 'en';
}
